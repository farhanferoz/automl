# FP-7 — zero-caller inventory + bootstrap/SE helper duplication

Completes the two items the budget-bounded duplication survey
(`docs/plans/capacity_programme/shared/reviews_2026-07-20/dup_inventory.md`, Deliverable 3 item 2 and
the Uncertainty section) left unfinished: (i) a full zero-caller sweep across every
`automl_package/examples/*.py` script (that pass sampled 9 router-related scripts + a handful of
architecture files, out of ~100 in the directory), and (ii) an inventory, with signatures, of the
suspected bootstrap/standard-error helper duplication that pass flagged but did not chase down.

**This is an inventory, not a cleanup.** No file under `automl_package/` was modified, moved, or
deleted by this task. Every `dead-candidate` disposition below is a nomination for a later task
(FP-8) to decide under its own three-part gate, per `docs/plans/capacity_programme/flexnn-package.md`
FP-7.b: "`dead-candidate` is a nomination, not a verdict."

## Method (verbatim from the FP-7.a mandate)

`automl_package/examples/` has no `__init__.py` — scripts there put the directory on `sys.path` and
import each other by bare module name (`import nested_width_net as nwn`), not via
`automl_package.examples.M`. A sweep that searches only the dotted-import form silently reports every
such live module as zero-caller. The required search form below covers all four reference shapes
(bare `import`/`from` statement, dotted-module reference, bare filename mention in prose/scripts, and
excludes only the module's own file) and was run **unmodified** for every one of the 104 modules in
`automl_package/examples/*.py`:

```bash
cd /home/ff235/dev/MLResearch/automl
M=<module_basename_without_py>
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+${M}\b)|(automl_package\.examples\.${M}\b)|(\b${M}\.py\b)" \
  --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' \
  automl_package/ docs/ tests/ \
  | grep -v "^automl_package/examples/${M}\.py:"
```

Every row below (all 104) records this exact command with `M` substituted, its output (verbatim
`<empty>`, or a match count + first N lines with a live example), whether the path is listed in
`docs/plans/capacity_programme/shared/PROTECTED.tsv`, and the resulting disposition.

## Part (i) — the full sweep

### Summary

| Disposition | Count |
|---|---|
| Total modules examined | 104 |
| `dead-candidate` (command printed nothing, not in PROTECTED.tsv) | 21 |
| `protected` (explicitly listed in PROTECTED.tsv — see reasons there; all 11 also have live callers per this sweep) | 11 |
| `live` (command printed at least one match, not in PROTECTED.tsv) | 72 |

21 + 11 + 72 = 104. No module was skipped.

### 🚨 Critical caveat found while cross-checking dead-candidates: 8 of the 21 have citations OUTSIDE the mandated search scope

The FP-7.a search form is scoped to exactly `automl_package/ docs/ tests/` (as specified). It does
**not** cover repository-root files. As a self-check (not part of the mandated command, done in
addition to it — see "Coverage boundary" below), I grepped each of the 21 `dead-candidate` module
names against `README.md CLAUDE.md RESUME.md CHANGELOG.md ARCHIVE.md SESSION_JOURNAL.md
SESSION_JOURNAL_2.md install.sh requirements.txt pyproject.toml ruff.toml` at the repo root. **8 of
the 21 are cited there:**

| Module | Root-doc citation |
|---|---|
| `run_automl` | `README.md:1173` — README's own example catalog: "`run_automl.py`: Demonstrates the core AutoML functionality for both regression and classification tasks." |
| `noisy_data_example` | `README.md:1174` — same catalog: "`noisy_data_example.py`: Provides a use case for comparing advanced regression models on noisy data." |
| `flexnn_probabilistic_test` | `CHANGELOG.md:573` |
| `probreg_ordering_ablation` | `CHANGELOG.md:474` (references its results directory, not the script directly — see per-module row) |
| `head_structure_diagnostic` | `SESSION_JOURNAL_2.md:81` |
| `multi_seed_sweep` | `SESSION_JOURNAL_2.md:87` |
| `probreg_mixture_eval` | `SESSION_JOURNAL_2.md:89` |
| `uci_benchmark` | `SESSION_JOURNAL_2.md:70` |

**This does not change the disposition recorded below** — `dead-candidate` is reported exactly as the
mandated command produced it, unmodified, per the task's own instruction to run that command exactly.
But it means 8 of the 21 rows carry an additional caveat flag (inline below) and none of the 21 should
be read as "confirmed safe to delete" from this document alone — `README.md`'s own example catalog
(two hits) is a particularly strong signal, since it is the project's front door for someone deciding
which example to run.

### All 21 `dead-candidate` rows

#### `automl_package/examples/aggregate_results.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+aggregate_results\\b)|(automl_package\\.examples\\.aggregate_results\\b)|(\\baggregate_results\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/aggregate_results\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

#### `automl_package/examples/capacity_ladder_variance_v0.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_variance_v0\\b)|(automl_package\\.examples\\.capacity_ladder_variance_v0\\b)|(\\bcapacity_ladder_variance_v0\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_variance_v0\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

#### `automl_package/examples/capacity_ladder_x2.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_x2\\b)|(automl_package\\.examples\\.capacity_ladder_x2\\b)|(\\bcapacity_ladder_x2\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_x2\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

#### `automl_package/examples/classifier_regression_nn_mapper_showcase.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+classifier_regression_nn_mapper_showcase\\b)|(automl_package\\.examples\\.classifier_regression_nn_mapper_showcase\\b)|(\\bclassifier_regression_nn_mapper_showcase\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/classifier_regression_nn_mapper_showcase\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

#### `automl_package/examples/dynamic_k_comparison.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+dynamic_k_comparison\\b)|(automl_package\\.examples\\.dynamic_k_comparison\\b)|(\\bdynamic_k_comparison\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/dynamic_k_comparison\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

#### `automl_package/examples/flexnn_probabilistic_test.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+flexnn_probabilistic_test\\b)|(automl_package\\.examples\\.flexnn_probabilistic_test\\b)|(\\bflexnn_probabilistic_test\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/flexnn_probabilistic_test\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

**CAVEAT — root-level doc citation OUTSIDE the FP-7.a search scope:** the mandated search
covers only `automl_package/ docs/ tests/`; it does not cover repo-root files. A separate
check of `README.md CLAUDE.md RESUME.md CHANGELOG.md ARCHIVE.md SESSION_JOURNAL.md
SESSION_JOURNAL_2.md install.sh requirements.txt pyproject.toml ruff.toml` found this module
cited at `CHANGELOG.md:573`. The dead-candidate disposition above is exactly what the required
search form produces and is reported unchanged; this citation does not overrule it -- it
is recorded so FP-8 does not treat this row as clear without also checking that citation.

#### `automl_package/examples/head_structure_diagnostic.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+head_structure_diagnostic\\b)|(automl_package\\.examples\\.head_structure_diagnostic\\b)|(\\bhead_structure_diagnostic\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/head_structure_diagnostic\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

**CAVEAT — root-level doc citation OUTSIDE the FP-7.a search scope:** the mandated search
covers only `automl_package/ docs/ tests/`; it does not cover repo-root files. A separate
check of `README.md CLAUDE.md RESUME.md CHANGELOG.md ARCHIVE.md SESSION_JOURNAL.md
SESSION_JOURNAL_2.md install.sh requirements.txt pyproject.toml ruff.toml` found this module
cited at `SESSION_JOURNAL_2.md:81`. The dead-candidate disposition above is exactly what the required
search form produces and is reported unchanged; this citation does not overrule it -- it
is recorded so FP-8 does not treat this row as clear without also checking that citation.

#### `automl_package/examples/house_price_prediction.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+house_price_prediction\\b)|(automl_package\\.examples\\.house_price_prediction\\b)|(\\bhouse_price_prediction\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/house_price_prediction\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

#### `automl_package/examples/independent_weights_showcase.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+independent_weights_showcase\\b)|(automl_package\\.examples\\.independent_weights_showcase\\b)|(\\bindependent_weights_showcase\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/independent_weights_showcase\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

#### `automl_package/examples/middle_class_centering.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+middle_class_centering\\b)|(automl_package\\.examples\\.middle_class_centering\\b)|(\\bmiddle_class_centering\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/middle_class_centering\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

#### `automl_package/examples/multi_seed_sweep.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+multi_seed_sweep\\b)|(automl_package\\.examples\\.multi_seed_sweep\\b)|(\\bmulti_seed_sweep\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/multi_seed_sweep\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

**CAVEAT — root-level doc citation OUTSIDE the FP-7.a search scope:** the mandated search
covers only `automl_package/ docs/ tests/`; it does not cover repo-root files. A separate
check of `README.md CLAUDE.md RESUME.md CHANGELOG.md ARCHIVE.md SESSION_JOURNAL.md
SESSION_JOURNAL_2.md install.sh requirements.txt pyproject.toml ruff.toml` found this module
cited at `SESSION_JOURNAL_2.md:87`. The dead-candidate disposition above is exactly what the required
search form produces and is reported unchanged; this citation does not overrule it -- it
is recorded so FP-8 does not treat this row as clear without also checking that citation.

#### `automl_package/examples/noisy_data_example.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+noisy_data_example\\b)|(automl_package\\.examples\\.noisy_data_example\\b)|(\\bnoisy_data_example\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/noisy_data_example\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

**CAVEAT — root-level doc citation OUTSIDE the FP-7.a search scope:** the mandated search
covers only `automl_package/ docs/ tests/`; it does not cover repo-root files. A separate
check of `README.md CLAUDE.md RESUME.md CHANGELOG.md ARCHIVE.md SESSION_JOURNAL.md
SESSION_JOURNAL_2.md install.sh requirements.txt pyproject.toml ruff.toml` found this module
cited at `README.md:1174`. The dead-candidate disposition above is exactly what the required
search form produces and is reported unchanged; this citation does not overrule it -- it
is recorded so FP-8 does not treat this row as clear without also checking that citation.

#### `automl_package/examples/overfitting_showcase.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+overfitting_showcase\\b)|(automl_package\\.examples\\.overfitting_showcase\\b)|(\\boverfitting_showcase\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/overfitting_showcase\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

#### `automl_package/examples/photo_z_domain_metrics_test.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+photo_z_domain_metrics_test\\b)|(automl_package\\.examples\\.photo_z_domain_metrics_test\\b)|(\\bphoto_z_domain_metrics_test\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/photo_z_domain_metrics_test\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

#### `automl_package/examples/probreg_k20_sweep.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_k20_sweep\\b)|(automl_package\\.examples\\.probreg_k20_sweep\\b)|(\\bprobreg_k20_sweep\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_k20_sweep\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

#### `automl_package/examples/probreg_mixture_eval.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_mixture_eval\\b)|(automl_package\\.examples\\.probreg_mixture_eval\\b)|(\\bprobreg_mixture_eval\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_mixture_eval\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

**CAVEAT — root-level doc citation OUTSIDE the FP-7.a search scope:** the mandated search
covers only `automl_package/ docs/ tests/`; it does not cover repo-root files. A separate
check of `README.md CLAUDE.md RESUME.md CHANGELOG.md ARCHIVE.md SESSION_JOURNAL.md
SESSION_JOURNAL_2.md install.sh requirements.txt pyproject.toml ruff.toml` found this module
cited at `SESSION_JOURNAL_2.md:89`. The dead-candidate disposition above is exactly what the required
search form produces and is reported unchanged; this citation does not overrule it -- it
is recorded so FP-8 does not treat this row as clear without also checking that citation.

#### `automl_package/examples/probreg_ordering_ablation.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_ordering_ablation\\b)|(automl_package\\.examples\\.probreg_ordering_ablation\\b)|(\\bprobreg_ordering_ablation\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_ordering_ablation\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

**CAVEAT — root-level doc citation OUTSIDE the FP-7.a search scope:** the mandated search
covers only `automl_package/ docs/ tests/`; it does not cover repo-root files. A separate
check of `README.md CLAUDE.md RESUME.md CHANGELOG.md ARCHIVE.md SESSION_JOURNAL.md
SESSION_JOURNAL_2.md install.sh requirements.txt pyproject.toml ruff.toml` found this module
cited at `CHANGELOG.md:474`. The dead-candidate disposition above is exactly what the required
search form produces and is reported unchanged; this citation does not overrule it -- it
is recorded so FP-8 does not treat this row as clear without also checking that citation.

#### `automl_package/examples/probreg_snr_sweep.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_snr_sweep\\b)|(automl_package\\.examples\\.probreg_snr_sweep\\b)|(\\bprobreg_snr_sweep\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_snr_sweep\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

#### `automl_package/examples/run_automl.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+run_automl\\b)|(automl_package\\.examples\\.run_automl\\b)|(\\brun_automl\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/run_automl\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

**CAVEAT — root-level doc citation OUTSIDE the FP-7.a search scope:** the mandated search
covers only `automl_package/ docs/ tests/`; it does not cover repo-root files. A separate
check of `README.md CLAUDE.md RESUME.md CHANGELOG.md ARCHIVE.md SESSION_JOURNAL.md
SESSION_JOURNAL_2.md install.sh requirements.txt pyproject.toml ruff.toml` found this module
cited at `README.md:1173`. The dead-candidate disposition above is exactly what the required
search form produces and is reported unchanged; this citation does not overrule it -- it
is recorded so FP-8 does not treat this row as clear without also checking that citation.

#### `automl_package/examples/toy_problem_pdf_reports.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+toy_problem_pdf_reports\\b)|(automl_package\\.examples\\.toy_problem_pdf_reports\\b)|(\\btoy_problem_pdf_reports\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/toy_problem_pdf_reports\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

#### `automl_package/examples/uci_benchmark.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+uci_benchmark\\b)|(automl_package\\.examples\\.uci_benchmark\\b)|(\\buci_benchmark\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/uci_benchmark\\.py:"
```
Output: `<empty>`
Protected (PROTECTED.tsv): no
Disposition: **dead-candidate**

**CAVEAT — root-level doc citation OUTSIDE the FP-7.a search scope:** the mandated search
covers only `automl_package/ docs/ tests/`; it does not cover repo-root files. A separate
check of `README.md CLAUDE.md RESUME.md CHANGELOG.md ARCHIVE.md SESSION_JOURNAL.md
SESSION_JOURNAL_2.md install.sh requirements.txt pyproject.toml ruff.toml` found this module
cited at `SESSION_JOURNAL_2.md:70`. The dead-candidate disposition above is exactly what the required
search form produces and is reported unchanged; this citation does not overrule it -- it
is recorded so FP-8 does not treat this row as clear without also checking that citation.

### All 11 `protected` rows

These 11 paths are listed in `docs/plans/capacity_programme/shared/PROTECTED.tsv` (whole-file or
symbol-level rows — see that file for the exact symbol and reason). Every one of them also has a
non-empty sweep result under this task's search form, consistent with PROTECTED.tsv's own claim that
these are live, bare-name-imported modules, not orphans that happen to be protected out of caution.

#### `automl_package/examples/capacity_ladder_k6.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_k6\\b)|(automl_package\\.examples\\.capacity_ladder_k6\\b)|(\\bcapacity_ladder_k6\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_k6\\.py:"
```
Output: 37 matches. First 3:
```
  automl_package/examples/capacity_ladder_s2.py:55:import capacity_ladder_k6 as ck6  # noqa: E402 — reuse _RouterMLP/read_table/_selftest_table/_jsonable verbatim
  automl_package/examples/capacity_ladder_s1.py:3:Successor to `capacity_ladder_k6.py`'s 3-arm HARD/SOFT/PILOT comparison. K6 established that
  automl_package/examples/capacity_ladder_s1.py:16:disk — no ladder retraining). Reuses `capacity_ladder_k6.py` verbatim for `_RouterMLP`,
```
Protected (PROTECTED.tsv): yes -- see PROTECTED.tsv rows for this path
Disposition: **protected**

#### `automl_package/examples/capacity_ladder_s1.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_s1\\b)|(automl_package\\.examples\\.capacity_ladder_s1\\b)|(\\bcapacity_ladder_s1\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_s1\\.py:"
```
Output: 17 matches. First 3:
```
  automl_package/examples/capacity_ladder_s2.py:3:S1 (`capacity_ladder_s1.py`) ran a target-construction FACTORIAL: every arm trains the router by
  automl_package/examples/capacity_ladder_s2.py:56:import capacity_ladder_s1 as cs1  # noqa: E402 — reuse _train_arm/_eval_arm/_oracle_reads/_paired_bootstrap_se/protocol constants
  automl_package/examples/capacity_ladder_results/RESULTS.md:416:  (`capacity_ladder_s1.py:273`, used by no bar/winner/diff) — corrected 2026-07-10 (`> 0`).
```
Protected (PROTECTED.tsv): yes -- see PROTECTED.tsv rows for this path
Disposition: **protected**

#### `automl_package/examples/capacity_ladder_s2.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_s2\\b)|(automl_package\\.examples\\.capacity_ladder_s2\\b)|(\\bcapacity_ladder_s2\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_s2\\.py:"
```
Output: 18 matches. First 3:
```
  automl_package/examples/capacity_ladder_results/RESULTS.md:463:Artifacts: `capacity_ladder_results/S2/{PREREGISTRATION.md,s2_summary.json}`, `capacity_ladder_s2.py`.
  automl_package/examples/capacity_ladder_results/S2/PREREGISTRATION.md:50:`automl_package/examples/capacity_ladder_s2.py` →
  automl_package/examples/capacity_ladder_results/S2/PREREGISTRATION.md:56:automl_package/examples/capacity_ladder_s2.py` (selftest first with `--selftest`).
```
Protected (PROTECTED.tsv): yes -- see PROTECTED.tsv rows for this path
Disposition: **protected**

#### `automl_package/examples/capacity_ladder_t2.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_t2\\b)|(automl_package\\.examples\\.capacity_ladder_t2\\b)|(\\bcapacity_ladder_t2\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_t2\\.py:"
```
Output: 25 matches. First 3:
```
  automl_package/examples/capacity_ladder_results/RESULTS.md:601:Artifacts: `capacity_ladder_results/T2/{PREREGISTRATION.md,t2_summary.json,T2_ADJUDICATION.md,nested_toyD_ndim_<config>_seed<seed>.pt}`, `capacity_ladder_t2.py`.
  automl_package/examples/capacity_ladder_results/T2/PREREGISTRATION.md:55:`automl_package/examples/capacity_ladder_t2.py` →
  automl_package/examples/capacity_ladder_results/T2/PREREGISTRATION.md:61:automl_package/examples/capacity_ladder_t2.py --selftest` first, then `--measure-one` (times ONE
```
Protected (PROTECTED.tsv): yes -- see PROTECTED.tsv rows for this path
Disposition: **protected**

#### `automl_package/examples/cascade_width_experiment.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+cascade_width_experiment\\b)|(automl_package\\.examples\\.cascade_width_experiment\\b)|(\\bcascade_width_experiment\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/cascade_width_experiment\\.py:"
```
Output: 16 matches. First 3:
```
  automl_package/examples/capacity_ladder_results/W_CASCADE/PREREGISTRATION.md:4:`cascade_width_experiment.py` (both arms), nets `cascade_width_net.py`/`matryoshka_width_net.py`.
  docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md:398:### 4.3 File 3 — `automl_package/examples/cascade_width_experiment.py` (new driver, covers BOTH arms)
  docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md:471:  AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/cascade_width_experiment.py --selftest
```
Protected (PROTECTED.tsv): yes -- see PROTECTED.tsv rows for this path
Disposition: **protected**

#### `automl_package/examples/cascade_width_net.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+cascade_width_net\\b)|(automl_package\\.examples\\.cascade_width_net\\b)|(\\bcascade_width_net\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/cascade_width_net\\.py:"
```
Output: 12 matches. First 3:
```
  automl_package/examples/cascade_width_experiment.py:56:import cascade_width_net as cwn  # noqa: E402
  automl_package/examples/capacity_ladder_results/W_CASCADE/PREREGISTRATION.md:4:`cascade_width_experiment.py` (both arms), nets `cascade_width_net.py`/`matryoshka_width_net.py`.
  docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md:327:### 4.1 File 1 — `automl_package/examples/cascade_width_net.py` (new)
```
Protected (PROTECTED.tsv): yes -- see PROTECTED.tsv rows for this path
Disposition: **protected**

#### `automl_package/examples/depth_composition_toy.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+depth_composition_toy\\b)|(automl_package\\.examples\\.depth_composition_toy\\b)|(\\bdepth_composition_toy\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/depth_composition_toy\\.py:"
```
Output: 71 matches. First 3:
```
  automl_package/examples/depth_selection_toy.py:91:from depth_composition_toy import (  # noqa: E402 — reuse the certified group/net building blocks, don't reinvent
  automl_package/examples/depth_selection_toy.py:98:from depth_composition_toy import train_clf as _train_clf_generic  # noqa: E402 — generic CE+convergence trainer, reused for the surface probe
  automl_package/examples/joint_capacity_toy.py:16:(`depth_composition_toy.py:295`): `state = tanh(block([state, input_t]))`, weight-shared across steps.
```
Protected (PROTECTED.tsv): yes -- see PROTECTED.tsv rows for this path
Disposition: **protected**

#### `automl_package/examples/depth_graded_toy.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+depth_graded_toy\\b)|(automl_package\\.examples\\.depth_graded_toy\\b)|(\\bdepth_graded_toy\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/depth_graded_toy\\.py:"
```
Output: 21 matches. First 3:
```
  automl_package/examples/depth_composition_toy.py:126:RUNG0_LR = 3e-3  # every certified A5/S5 L=10 run uses this; depth_graded_toy.py:74 -- "1e-2 stalls the deep unroll, 3e-3 reaches 0.99"
  automl_package/examples/depth_selection_toy.py:99:from depth_graded_toy import CHECK_EVERY, MIN_DELTA, PATIENCE  # noqa: E402 — shared convergence-gate hyperparameter convention (anytime-net training loop)
  docs/plans/capacity_programme/width-depth.md:9:  (`automl_package/examples/depth_graded_toy.py`). "Cross the two toys" is therefore a DESIGN
```
Protected (PROTECTED.tsv): yes -- see PROTECTED.tsv rows for this path
Disposition: **protected**

#### `automl_package/examples/matryoshka_width_net.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+matryoshka_width_net\\b)|(automl_package\\.examples\\.matryoshka_width_net\\b)|(\\bmatryoshka_width_net\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/matryoshka_width_net\\.py:"
```
Output: 13 matches. First 3:
```
  automl_package/examples/cascade_width_experiment.py:59:import matryoshka_width_net as mwn  # noqa: E402
  automl_package/examples/capacity_ladder_results/W_CASCADE/PREREGISTRATION.md:4:`cascade_width_experiment.py` (both arms), nets `cascade_width_net.py`/`matryoshka_width_net.py`.
  docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md:375:### 4.2 File 2 — `automl_package/examples/matryoshka_width_net.py` (new)
```
Protected (PROTECTED.tsv): yes -- see PROTECTED.tsv rows for this path
Disposition: **protected**

#### `automl_package/examples/nested_width_net.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+nested_width_net\\b)|(automl_package\\.examples\\.nested_width_net\\b)|(\\bnested_width_net\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/nested_width_net\\.py:"
```
Output: 99 matches. First 3:
```
  automl_package/examples/cascade_width_net.py:57:import nested_width_net as nwn  # noqa: E402
  automl_package/examples/moe_flexnn_comparison.py:80:import nested_width_net as nwn  # noqa: E402
  automl_package/examples/matryoshka_width_net.py:52:import nested_width_net as nwn  # noqa: E402
```
Protected (PROTECTED.tsv): yes -- see PROTECTED.tsv rows for this path
Disposition: **protected**

#### `automl_package/examples/sinc_width_experiment.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+sinc_width_experiment\\b)|(automl_package\\.examples\\.sinc_width_experiment\\b)|(\\bsinc_width_experiment\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/sinc_width_experiment\\.py:"
```
Output: 79 matches. First 3:
```
  automl_package/examples/hetero_width_experiment.py:6:W1 (`sinc_width_experiment.py`) FAILED bar (i) CONSTRUCTION on the ramped-sinc toy for two reasons
  automl_package/examples/hetero_width_experiment.py:17:Otherwise this driver is a straight mirror of `sinc_width_experiment.py`: same two-stage/no-leak
  automl_package/examples/hetero_width_experiment.py:53:import sinc_width_experiment as sw  # noqa: E402 — reuse RunConfig + the 3 bar functions + score/selector helpers verbatim
```
Protected (PROTECTED.tsv): yes -- see PROTECTED.tsv rows for this path
Disposition: **protected**

### All 72 `live` rows

Non-empty sweep result, not listed in PROTECTED.tsv. These are ordinary live modules — no action
implied; listed for completeness since FP-7.b requires every candidate's command, output, protection
status, and disposition, not only the dead ones.

#### `automl_package/examples/ablation_study.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+ablation_study\\b)|(automl_package\\.examples\\.ablation_study\\b)|(\\bablation_study\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/ablation_study\\.py:"
```
Output: 1 matches. First 3:
```
  automl_package/examples/run_all_sweeps.sh:100:run_sweep "ablation_study"    "automl_package.examples.ablation_study"
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_accounting.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_accounting\\b)|(automl_package\\.examples\\.capacity_accounting\\b)|(\\bcapacity_accounting\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_accounting\\.py:"
```
Output: 69 matches. First 3:
```
  automl_package/examples/moe_regression.py:32:this task's file list (`capacity_accounting.py` is owned by a concurrent capacity-programme task
  automl_package/examples/moe_regression.py:69:import capacity_accounting as ca  # noqa: E402
  automl_package/examples/moe_regression.py:273:    `task={mse,ce}` section for why this lives here instead of in `capacity_accounting.py`.
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/_capacity_ladder.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+_capacity_ladder\\b)|(automl_package\\.examples\\._capacity_ladder\\b)|(\\b_capacity_ladder\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/_capacity_ladder\\.py:"
```
Output: 33 matches. First 3:
```
  automl_package/examples/capacity_ladder_f1_validation.py:50:import _capacity_ladder as cladder  # noqa: E402
  automl_package/examples/capacity_ladder_f1_validation.py:311:    bootstrap/abstain/saturation behavior is validated by `_capacity_ladder.py`'s own selftest, not
  automl_package/examples/capacity_ladder_k4.py:43:import _capacity_ladder as cl
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_f1_validation.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_f1_validation\\b)|(automl_package\\.examples\\.capacity_ladder_f1_validation\\b)|(\\bcapacity_ladder_f1_validation\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_f1_validation\\.py:"
```
Output: 3 matches. First 3:
```
  automl_package/examples/capacity_ladder_f2.py:13:activation) are the same as `capacity_ladder_f1_validation.py`'s fixed-capacity MLP sweep;
  automl_package/examples/capacity_ladder_f2.py:78:# F1's fixed hyperparameters (`capacity_ladder_f1_validation.py`), reused verbatim; only
  docs/plans/capacity_programme/shared/reviews_2026-07-20/audit_api.md:43:- **Depth, expensive-sweep**: `run_one_case`'s fixed-depth loop (automl_package/examples/capacity_ladder_f2.py:294-300): `for depth in range(1, cfg.max_depth+1): fixed_model = _build_model(FlexibleHiddenLayersNN, depth, LayerSelectionMethod.NONE, ...); fixed_model.fit(...)`. Genuine train-N-separate-models sweep using the ACTUAL package class. SCRIPT. (Also `capacity_ladder_f1_validation.py:127-198` sweeps depths 1..6 but with a bespoke local `_MLP` class, NOT `FlexibleHiddenLayersNN`/`IndependentWeightsFlexibleNN` — noted for completeness but not the primary citation since it bypasses the package model classes entirely.)
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_f2.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_f2\\b)|(automl_package\\.examples\\.capacity_ladder_f2\\b)|(\\bcapacity_ladder_f2\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_f2\\.py:"
```
Output: 24 matches. First 3:
```
  automl_package/examples/capacity_ladder_x1.py:268:        print("no F2 tables available -- run capacity_ladder_f2.py first.")
  automl_package/examples/capacity_ladder_t3.py:89:    `capacity_ladder_f2.py`/`f3.py`/`x1.py`/`k1k2k3.py`'s `_boot_se`/`_plain_boot_se`.
  automl_package/examples/capacity_ladder_p1.py:10:Nature: retrains the F2 nested-depth ladder (`capacity_ladder_f2.py`'s `_build_model` +
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_f3.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_f3\\b)|(automl_package\\.examples\\.capacity_ladder_f3\\b)|(\\bcapacity_ladder_f3\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_f3\\.py:"
```
Output: 9 matches. First 3:
```
  automl_package/examples/capacity_ladder_t1.py:84:import capacity_ladder_f3 as f3mod  # noqa: E402  (load_f2_table, for the G-flat control band)
  automl_package/examples/capacity_ladder_p1.py:64:from capacity_ladder_f3 import _jsonable, load_f2_table  # noqa: E402
  automl_package/examples/capacity_ladder_x1.py:46:from capacity_ladder_f3 import SEEDS, TOYS, _jsonable, load_f2_table  # noqa: E402
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_h1.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_h1\\b)|(automl_package\\.examples\\.capacity_ladder_h1\\b)|(\\bcapacity_ladder_h1\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_h1\\.py:"
```
Output: 29 matches. First 3:
```
  automl_package/examples/nested_width_net.py:28:and `capacity_ladder_h1.py::_train_phase1`'s full-batch draw-per-epoch loop shape. No selector
  automl_package/examples/nested_width_net.py:451:    semantics (layer_selection_strategies.py:187-202) and `capacity_ladder_h1.py::_train_phase1`'s
  automl_package/examples/capacity_ladder_results/RESULTS.md:657:`capacity_ladder_h1.py`.
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_k1k2k3.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_k1k2k3\\b)|(automl_package\\.examples\\.capacity_ladder_k1k2k3\\b)|(\\bcapacity_ladder_k1k2k3\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_k1k2k3\\.py:"
```
Output: 2 matches. First 3:
```
  automl_package/examples/capacity_ladder_f2.py:202:# same convention as `capacity_ladder_k1k2k3.py`'s `_plain_boot_se`, not reimplemented.
  automl_package/examples/capacity_ladder_f3.py:9:K1/K2 ran on the WS1 score tables (`capacity_ladder_k1k2k3.py`), on those F2 tables instead,
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_k4.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_k4\\b)|(automl_package\\.examples\\.capacity_ladder_k4\\b)|(\\bcapacity_ladder_k4\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_k4\\.py:"
```
Output: 5 matches. First 3:
```
  automl_package/examples/capacity_ladder_x4.py:6:K4 (`capacity_ladder_k4.py`) trains a NESTED-k surrogate on toy E (moving modes: two components
  automl_package/examples/capacity_ladder_results/R2_verdict.md:7:(`_capacity_ladder.py`, `_capacity_ladder_nested.py`, `capacity_ladder_k4.py`,
  automl_package/examples/capacity_ladder_results/R2_verdict.md:102:(`capacity_ladder_k4.py:137-147`). Verified:
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_k5.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_k5\\b)|(automl_package\\.examples\\.capacity_ladder_k5\\b)|(\\bcapacity_ladder_k5\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_k5\\.py:"
```
Output: 6 matches. First 3:
```
  automl_package/examples/capacity_ladder_k6.py:4:recovered on toy D via the held-out ARBITER (`capacity_ladder_k5.py`'s neighbour-averaged
  automl_package/examples/capacity_ladder_k6.py:29:(missing files are skipped, matching `capacity_ladder_k5.py`'s `read_case`).
  automl_package/examples/capacity_ladder_k6.py:54:import capacity_ladder_k5 as ck5  # noqa: E402 — reuse perinput_knee_curve verbatim (the K5 read)
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/_capacity_ladder_nested.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+_capacity_ladder_nested\\b)|(automl_package\\.examples\\._capacity_ladder_nested\\b)|(\\b_capacity_ladder_nested\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/_capacity_ladder_nested\\.py:"
```
Output: 23 matches. First 3:
```
  automl_package/enums.py:133:    from `automl_package/examples/_capacity_ladder_nested.py`), with per-input k at inference
  automl_package/examples/capacity_ladder_k4.py:44:import _capacity_ladder_nested as ckn
  automl_package/examples/capacity_ladder_h1.py:67:import _variational_em as vem  # noqa: E402 — reuse gaussian_log_density verbatim (matches _capacity_ladder_nested.py)
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_p1.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_p1\\b)|(automl_package\\.examples\\.capacity_ladder_p1\\b)|(\\bcapacity_ladder_p1\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_p1\\.py:"
```
Output: 3 matches. First 3:
```
  automl_package/examples/capacity_ladder_results/RESULTS.md:550:Artifacts: `capacity_ladder_results/P1/{PREREGISTRATION.md,p1_summary.json,p1_summary_N500.json,p1_summary_N2000.json,p1_summary_N8000.json,P1_ADJUDICATION.md}`, `capacity_ladder_p1.py`.
  automl_package/examples/capacity_ladder_results/P1/P1_ADJUDICATION.md:189:Artifacts: `capacity_ladder_results/P1/{PREREGISTRATION.md,p1_summary.json,p1_summary_N500.json,p1_summary_N2000.json,p1_summary_N8000.json,P1_ADJUDICATION.md}`, `capacity_ladder_p1.py`.
  docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md:352:below its own floor at every N. **Script:** `capacity_ladder_p1.py` →
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_t1.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_t1\\b)|(automl_package\\.examples\\.capacity_ladder_t1\\b)|(\\bcapacity_ladder_t1\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_t1\\.py:"
```
Output: 14 matches. First 3:
```
  automl_package/examples/capacity_ladder_t1_path2.py:56:from capacity_ladder_t1 import (  # noqa: E402
  automl_package/examples/capacity_ladder_t1_path1.py:9:the SINGLE-restart config used by `capacity_ladder_t1.py` cannot, by itself, distinguish "tent^5 is
  automl_package/examples/capacity_ladder_t1_path1.py:19:Data are byte-identical to `capacity_ladder_t1.py`'s seed-0 case (`make_toy_t1(n=N_TRAIN, seed=0)`
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_t1_path1.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_t1_path1\\b)|(automl_package\\.examples\\.capacity_ladder_t1_path1\\b)|(\\bcapacity_ladder_t1_path1\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_t1_path1\\.py:"
```
Output: 12 matches. First 3:
```
  automl_package/examples/capacity_ladder_t1_path2.py:66:from capacity_ladder_t1_path1 import _UNLEARNABLE_GAP_NAT, N_RESTARTS, ORACLE_LL, SIGMA  # noqa: E402
  automl_package/examples/capacity_ladder_results/RESULTS.md:729:T1_PATH2_ADJUDICATION.md}`, `capacity_ladder_t1.py`, `capacity_ladder_t1_path1.py`, `capacity_ladder_t1_path2.py`.
  automl_package/examples/capacity_ladder_results/T1/T1_PATH1_ADJUDICATION.md:3:Fresh-context Opus adjudicator. The path-1 DRIVER (`capacity_ladder_t1_path1.py`) was newly
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_t1_path2.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_t1_path2\\b)|(automl_package\\.examples\\.capacity_ladder_t1_path2\\b)|(\\bcapacity_ladder_t1_path2\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_t1_path2\\.py:"
```
Output: 6 matches. First 3:
```
  automl_package/examples/capacity_ladder_results/RESULTS.md:707:  `capacity_ladder_t1_path2.py`). **All three read `NOT_FOUND_UNLEARNABLE`** (0/3 construction_pass each;
  automl_package/examples/capacity_ladder_results/RESULTS.md:729:T1_PATH2_ADJUDICATION.md}`, `capacity_ladder_t1.py`, `capacity_ladder_t1_path1.py`, `capacity_ladder_t1_path2.py`.
  automl_package/examples/capacity_ladder_results/T1/T1_PATH2_ADJUDICATION.md:22:and the per-seed loop (`capacity_ladder_t1_path2.py`).
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_t3.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_t3\\b)|(automl_package\\.examples\\.capacity_ladder_t3\\b)|(\\bcapacity_ladder_t3\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_t3\\.py:"
```
Output: 10 matches. First 3:
```
  automl_package/examples/capacity_ladder_results/RESULTS.md:494:  `x4b.fit_and_score_seed_mr` VERBATIM (`capacity_ladder_t3.py:137`), so the N=1000 cell *reproduces* X4b
  automl_package/examples/capacity_ladder_results/RESULTS.md:506:Artifacts: `capacity_ladder_results/T3/{PREREGISTRATION.md,t3_summary.json}`, `capacity_ladder_t3.py`.
  automl_package/examples/capacity_ladder_results/T3/PREREGISTRATION.md:17:question being asked. `capacity_ladder_t3.py` calls `x4b.fit_and_score_seed_mr` and
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/_capacity_ladder_toys.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+_capacity_ladder_toys\\b)|(automl_package\\.examples\\._capacity_ladder_toys\\b)|(\\b_capacity_ladder_toys\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/_capacity_ladder_toys\\.py:"
```
Output: 16 matches. First 3:
```
  automl_package/examples/capacity_ladder_f2.py:6:G / G-flat / H (`_capacity_ladder_toys.py`, validated by F1): per-sample depth
  automl_package/examples/capacity_ladder_f2.py:64:import _capacity_ladder_toys as toys  # noqa: E402
  automl_package/examples/capacity_ladder_f1_validation.py:51:import _capacity_ladder_toys as toys  # noqa: E402
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_v2.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_v2\\b)|(automl_package\\.examples\\.capacity_ladder_v2\\b)|(\\bcapacity_ladder_v2\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_v2\\.py:"
```
Output: 3 matches. First 3:
```
  automl_package/examples/capacity_ladder_x2.py:52:import capacity_ladder_v2 as v2  # noqa: E402
  automl_package/examples/capacity_ladder_v3.py:69:import capacity_ladder_v2 as v2  # noqa: E402
  automl_package/examples/capacity_ladder_results/V2/V2_findings.md:3:Source: `capacity_ladder_v2.py` on V-toy1 (1-D heteroscedastic, known smooth σ(x)).
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_v3.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_v3\\b)|(automl_package\\.examples\\.capacity_ladder_v3\\b)|(\\bcapacity_ladder_v3\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_v3\\.py:"
```
Output: 1 matches. First 3:
```
  automl_package/examples/capacity_ladder_x2.py:53:import capacity_ladder_v3 as v3  # noqa: E402
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_variance_v1.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_variance_v1\\b)|(automl_package\\.examples\\.capacity_ladder_variance_v1\\b)|(\\bcapacity_ladder_variance_v1\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_variance_v1\\.py:"
```
Output: 4 matches. First 3:
```
  automl_package/examples/capacity_ladder_v3.py:70:import capacity_ladder_variance_v1 as v1  # noqa: E402
  automl_package/examples/capacity_ladder_v2.py:58:import capacity_ladder_variance_v1 as v1
  automl_package/examples/capacity_ladder_x2.py:54:import capacity_ladder_variance_v1 as v1  # noqa: E402
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_x1.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_x1\\b)|(automl_package\\.examples\\.capacity_ladder_x1\\b)|(\\bcapacity_ladder_x1\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_x1\\.py:"
```
Output: 3 matches. First 3:
```
  automl_package/examples/capacity_ladder_t1.py:85:import capacity_ladder_x1 as x1mod  # noqa: E402  (hierarchical_perbin_stack, run_repeated)
  automl_package/examples/capacity_ladder_results/T1/_bar_ii_probe.py:21:import capacity_ladder_x1 as x1mod  # noqa: E402
  docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md:216:(50 splits, Nadeau–Bengio SE) + `capacity_ladder_x1.py` `hierarchical_perbin_stack`
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_x3.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_x3\\b)|(automl_package\\.examples\\.capacity_ladder_x3\\b)|(\\bcapacity_ladder_x3\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_x3\\.py:"
```
Output: 5 matches. First 3:
```
  automl_package/examples/capacity_ladder_t1.py:86:import capacity_ladder_x3 as x3mod  # noqa: E402  (run_repeated_crossfit)
  automl_package/examples/capacity_ladder_f3.py:183:    original F3 single-split read is reproduced bit-identically. X3 (`capacity_ladder_x3.py`) sweeps it
  automl_package/examples/capacity_ladder_p1.py:65:from capacity_ladder_x3 import _pooled_across_seeds, _synthetic_table, run_repeated_crossfit  # noqa: E402
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_x4.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_x4\\b)|(automl_package\\.examples\\.capacity_ladder_x4\\b)|(\\bcapacity_ladder_x4\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_x4\\.py:"
```
Output: 4 matches. First 3:
```
  automl_package/examples/capacity_ladder_t3.py:63:import capacity_ladder_x4 as x4  # noqa: E402 — reuse TOY_SPECS + tercile/verdict machinery verbatim
  automl_package/examples/capacity_ladder_x4b.py:4: capacity_ladder_x4.py; EXECUTION_PLAN §8.5 X4 follow-up / RESULTS.md "## X-queue follow-ups".)
  automl_package/examples/capacity_ladder_x4b.py:55:import capacity_ladder_x4 as x4  # noqa: E402  — reuse the parent task's helpers verbatim
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/capacity_ladder_x4b.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+capacity_ladder_x4b\\b)|(automl_package\\.examples\\.capacity_ladder_x4b\\b)|(\\bcapacity_ladder_x4b\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/capacity_ladder_x4b\\.py:"
```
Output: 6 matches. First 3:
```
  automl_package/examples/capacity_ladder_t3.py:7:non-nested arbiter (`capacity_ladder_x4b.py`) reads model-capture 2/3 or 3/3 depending on seed and
  automl_package/examples/capacity_ladder_t3.py:13:Instrument (reused VERBATIM, not reimplemented): `capacity_ladder_x4b.py`'s E-lane non-nested
  automl_package/examples/capacity_ladder_t3.py:64:import capacity_ladder_x4b as x4b  # noqa: E402 — reuse fit_and_score_seed_mr (R=8 multi-restart) verbatim
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/classifier_symmetry_check.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+classifier_symmetry_check\\b)|(automl_package\\.examples\\.classifier_symmetry_check\\b)|(\\bclassifier_symmetry_check\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/classifier_symmetry_check\\.py:"
```
Output: 4 matches. First 3:
```
  docs/probreg_identifiability_implementation_plan.md:8:Symmetry-check diagnostic (`automl_package/examples/classifier_symmetry_check.py`)
  docs/probreg_identifiability_implementation_plan.md:414:`classifier_symmetry_check.py` (heteroscedastic, bimodal, piecewise, exponential).
  docs/probreg_identifiability_implementation_plan.md:421:(dataset, k, seed). Same config as `classifier_symmetry_check.py`: 80 epochs,
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/classreg_k_sweep.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+classreg_k_sweep\\b)|(automl_package\\.examples\\.classreg_k_sweep\\b)|(\\bclassreg_k_sweep\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/classreg_k_sweep\\.py:"
```
Output: 2 matches. First 3:
```
  automl_package/examples/run_all_sweeps.sh:96:run_sweep "classreg_k"        "automl_package.examples.classreg_k_sweep"
  docs/phase2_handover.md:142:with P2.3 once P2.1 passes. Existing `classreg_k_sweep.py` has single-seed
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/classreg_probability_sanity.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+classreg_probability_sanity\\b)|(automl_package\\.examples\\.classreg_probability_sanity\\b)|(\\bclassreg_probability_sanity\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/classreg_probability_sanity\\.py:"
```
Output: 2 matches. First 3:
```
  docs/probreg_identifiability_implementation_plan.md:27:`automl_package/examples/classreg_probability_sanity.py`): classifier p_i(x)
  docs/probreg_identifiability_implementation_plan.md:502:     (as in `classreg_probability_sanity.py`)
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/converged_width_experiment.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+converged_width_experiment\\b)|(automl_package\\.examples\\.converged_width_experiment\\b)|(\\bconverged_width_experiment\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/converged_width_experiment\\.py:"
```
Output: 23 matches. First 3:
```
  automl_package/examples/cascade_width_experiment.py:6:Runs BOTH arms on the identical data/split/seeds as `converged_width_experiment.py` (reused
  automl_package/examples/cascade_width_experiment.py:57:import converged_width_experiment as cwe  # noqa: E402
  automl_package/examples/cascade_width_experiment.py:70:# Regime reused verbatim from converged_width_experiment.py so every summary is directly comparable.
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/convergence.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+convergence\\b)|(automl_package\\.examples\\.convergence\\b)|(\\bconvergence\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/convergence\\.py:"
```
Output: 61 matches. First 3:
```
  automl_package/examples/matryoshka_width_net.py:51:import convergence as cvg  # noqa: E402
  automl_package/examples/cascade_width_experiment.py:58:import convergence as cvg  # noqa: E402
  automl_package/examples/kdropout_converged_width_experiment.py:8:per-width convergence rule (`convergence.py`; agent-memory
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/depth_selection_toy.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+depth_selection_toy\\b)|(automl_package\\.examples\\.depth_selection_toy\\b)|(\\bdepth_selection_toy\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/depth_selection_toy\\.py:"
```
Output: 75 matches. First 3:
```
  automl_package/examples/depth_composition_toy.py:85:# (`depth_selection_toy.py`, D8b): four ORDER-2 involutions (double-transpositions), verified below to
  automl_package/examples/depth_composition_toy.py:127:RUNG0_CLIP_MAX_NORM = 1.0  # depth_selection_toy.py:114 -- "L=10 needs clipping to stay GD-trainable"
  automl_package/examples/depth_composition_toy.py:164:    # cross-caller constraint (`depth_selection_toy.py:570`) requires this to stay the default.
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/depth_toy.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+depth_toy\\b)|(automl_package\\.examples\\.depth_toy\\b)|(\\bdepth_toy\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/depth_toy\\.py:"
```
Output: 6 matches. First 3:
```
  docs/plans/capacity_programme/shared/reviews_2026-07-20/audit_api.md:44:- **Width, expensive-sweep**: NOT FOUND using the package's `FlexibleWidthNN` class. `sinc_width_experiment.py` (which does a width sweep) imports `nested_width_net as nwn` (automl_package/examples/sinc_width_experiment.py:68) — its OWN research module's `NestedWidthNet`/`IndependentWidthNet`, not `automl_package.models.flexible_width_network.FlexibleWidthNN`. Grepped `grep -rn "for width in\|WIDTHS\s*=" automl_package/examples/*.py`: only hits in `capacity_ladder_s1.py:471` and `depth_toy.py:752`, neither imports `FlexibleWidthNN`. Provisionally ABSENT for width-family expensive-sweep against the real package class — grepped `grep -rln "FlexibleWidthNN(" automl_package/ tests/` → only 3 files (flexible_width_network.py itself, tests/test_flexible_width_network.py, moe_flexnn_comparison.py), and moe_flexnn_comparison.py's usage is the "cheap global" pattern below, not a per-width retrain sweep.
  docs/depth_capacity/depth_toy_negative_note.md:11:easy-region-flat — thresholds and formulas in `automl_package/examples/depth_toy.py`'s module
  docs/depth_capacity/depth_toy_negative_note.md:100:- `automl_package/examples/depth_toy.py` — generators (all 3 candidates) + net builders + probe
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/exponential_symlog_probe.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+exponential_symlog_probe\\b)|(automl_package\\.examples\\.exponential_symlog_probe\\b)|(\\bexponential_symlog_probe\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/exponential_symlog_probe\\.py:"
```
Output: 2 matches. First 3:
```
  docs/probreg_identifiability_research.md:824:Script: `automl_package/examples/exponential_symlog_probe.py`.
  docs/probreg_identifiability_research.md:1159:| symlog probe | `automl_package/examples/exponential_symlog_probe.py` |
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/final_results_report.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+final_results_report\\b)|(automl_package\\.examples\\.final_results_report\\b)|(\\bfinal_results_report\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/final_results_report\\.py:"
```
Output: 3 matches. First 3:
```
  automl_package/examples/run_all_sweeps.sh:108:"${PY}" -m automl_package.examples.final_results_report \
  docs/probreg_identifiability_research.md:1095:## 10.4 `final_results_report.py` column name (`fe63c90`)
  docs/probreg_identifiability_research.md:1160:| aggregator PDF | `automl_package/examples/final_results_report.py` |
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/flexible_nn_showcase.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+flexible_nn_showcase\\b)|(automl_package\\.examples\\.flexible_nn_showcase\\b)|(\\bflexible_nn_showcase\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/flexible_nn_showcase\\.py:"
```
Output: 1 matches. First 3:
```
  docs/architecture_analysis.md:428:### 4.2 `flexible_nn_showcase.py` — GOOD
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/flex_nn_ablation.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+flex_nn_ablation\\b)|(automl_package\\.examples\\.flex_nn_ablation\\b)|(\\bflex_nn_ablation\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/flex_nn_ablation\\.py:"
```
Output: 1 matches. First 3:
```
  automl_package/examples/run_all_sweeps.sh:99:run_sweep "flex_nn_ablation"  "automl_package.examples.flex_nn_ablation"
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/flex_nn_depth_viz.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+flex_nn_depth_viz\\b)|(automl_package\\.examples\\.flex_nn_depth_viz\\b)|(\\bflex_nn_depth_viz\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/flex_nn_depth_viz\\.py:"
```
Output: 2 matches. First 3:
```
  automl_package/examples/flexnn_revalidation.py:12:DepthRegularization\\|n_predictor" automl_package/examples/` surfaced `flex_nn_depth_viz.py` (I3),
  automl_package/examples/flexnn_revalidation.py:42:from automl_package.examples.flex_nn_depth_viz import extract_depth, piecewise_dataset
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/_flexnn_prefix_selftest.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+_flexnn_prefix_selftest\\b)|(automl_package\\.examples\\._flexnn_prefix_selftest\\b)|(\\b_flexnn_prefix_selftest\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/_flexnn_prefix_selftest\\.py:"
```
Output: 3 matches. First 3:
```
  automl_package/examples/nested_width_net.py:39:      to both widths (the nested-prefix property, mirroring `_flexnn_prefix_selftest.py`'s
  automl_package/models/flexnn/strategies/layer.py:199:    property this relies on is the one audited in `_flexnn_prefix_selftest.py`.
  tests/test_nested_depth_strategy.py:7:(`automl_package/examples/_flexnn_prefix_selftest.py`) through the NESTED path instead
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/flexnn_revalidation.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+flexnn_revalidation\\b)|(automl_package\\.examples\\.flexnn_revalidation\\b)|(\\bflexnn_revalidation\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/flexnn_revalidation\\.py:"
```
Output: 4 matches. First 3:
```
  automl_package/examples/moe_flexnn_comparison.py:36:(`automl_package/examples/flexnn_revalidation.py`), not a new convention. MoE training reuses
  automl_package/examples/moe_flexnn_comparison.py:233:# directly, replay the trajectory through ConvergenceTracker (flexnn_revalidation.py pattern).
  automl_package/examples/report_a_benchmark.py:44:`flexnn_revalidation.py` uses (`hit_cap = len(val_loss_history) >= N_EPOCHS_CAP`). A cell with
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/full_benchmark.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+full_benchmark\\b)|(automl_package\\.examples\\.full_benchmark\\b)|(\\bfull_benchmark\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/full_benchmark\\.py:"
```
Output: 5 matches. First 3:
```
  automl_package/examples/run_all_sweeps.sh:97:run_sweep "full_benchmark"    "automl_package.examples.full_benchmark"
  automl_package/examples/report_a_benchmark.py:9:documented best combo), `automl_package/examples/full_benchmark.py` (the toy suite:
  automl_package/examples/report_a_benchmark.py:84:from automl_package.examples.full_benchmark import _exponential, _heteroscedastic, _multimodal, _piecewise
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/gumbel_elbo_retest.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+gumbel_elbo_retest\\b)|(automl_package\\.examples\\.gumbel_elbo_retest\\b)|(\\bgumbel_elbo_retest\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/gumbel_elbo_retest\\.py:"
```
Output: 1 matches. First 3:
```
  automl_package/examples/flexnn_revalidation.py:14:`gumbel_elbo_retest.py` (I8), a post-fix NONE-vs-ELBO retest. Neither meets this task's spec (no
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/head_degeneracy_diagnostic.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+head_degeneracy_diagnostic\\b)|(automl_package\\.examples\\.head_degeneracy_diagnostic\\b)|(\\bhead_degeneracy_diagnostic\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/head_degeneracy_diagnostic\\.py:"
```
Output: 1 matches. First 3:
```
  docs/probreg_identifiability_implementation_plan.md:103:`head_degeneracy_diagnostic.py` no longer produces MSE > 1.0 on the exponential
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/hetero_width_experiment.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+hetero_width_experiment\\b)|(automl_package\\.examples\\.hetero_width_experiment\\b)|(\\bhetero_width_experiment\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/hetero_width_experiment\\.py:"
```
Output: 15 matches. First 3:
```
  automl_package/examples/independent_width_experiment.py:7:W1/W2 (`sinc_width_experiment.py` / `hetero_width_experiment.py`) fail bar (i) because the SHARED
  automl_package/examples/independent_width_experiment.py:16:Everything else mirrors `hetero_width_experiment.py` verbatim: same `make_hetero` toy, same
  automl_package/examples/independent_width_experiment.py:179:    import hetero_width_experiment as hw  # noqa: PLC0415 — selftest-only reuse of the planted-table bar wiring
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/hpo_sweep.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+hpo_sweep\\b)|(automl_package\\.examples\\.hpo_sweep\\b)|(\\bhpo_sweep\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/hpo_sweep\\.py:"
```
Output: 1 matches. First 3:
```
  automl_package/examples/run_all_sweeps.sh:101:run_sweep "hpo"               "automl_package.examples.hpo_sweep"  "${ALL_DATASETS:+ALL_DATASETS=1}"
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/independent_width_experiment.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+independent_width_experiment\\b)|(automl_package\\.examples\\.independent_width_experiment\\b)|(\\bindependent_width_experiment\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/independent_width_experiment\\.py:"
```
Output: 5 matches. First 3:
```
  automl_package/examples/converged_width_experiment.py:3:This replaces the fixed-epoch width runs (`independent_width_experiment.py` and the scratchpad probes)
  docs/width_dial_synthesis_2026-07-13/per_input_width_architecture_readthrough.md:629:| `W_INDEP` | `independent_width_experiment.py` | `IndependentWidthNet` | K× | 2.5k ep | mixed (1.37/1.55/1.51) | 3/3 direction |
  docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md:53:| `W_INDEP` | `independent_width_experiment.py` | `IndependentWidthNet` | K× | 2.5k ep | mixed (1.37/1.55/1.51) | 3/3 direction |
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/investigation_pdf_reports.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+investigation_pdf_reports\\b)|(automl_package\\.examples\\.investigation_pdf_reports\\b)|(\\binvestigation_pdf_reports\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/investigation_pdf_reports\\.py:"
```
Output: 1 matches. First 3:
```
  automl_package/examples/final_results_report.py:53:# PDF helpers (mirrors automl_package.examples.investigation_pdf_reports)
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/joint_capacity_toy.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+joint_capacity_toy\\b)|(automl_package\\.examples\\.joint_capacity_toy\\b)|(\\bjoint_capacity_toy\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/joint_capacity_toy\\.py:"
```
Output: 10 matches. First 3:
```
  docs/plans/capacity_programme/width-depth.md:58:      artifacts:** implement the toy as `automl_package/examples/joint_capacity_toy.py` (reusing
  docs/plans/capacity_programme/width-depth.md:98:**What ran.** `automl_package/examples/joint_capacity_toy.py` built (J-1 readout + J-2 block), verified
  docs/plans/capacity_programme/shared/reviews_2026-07-20/audit_crosscut.md:126:**What ran:** `automl_package/examples/joint_capacity_toy.py`, two candidate constructions (J-1
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/kdropout_converged_width_experiment.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+kdropout_converged_width_experiment\\b)|(automl_package\\.examples\\.kdropout_converged_width_experiment\\b)|(\\bkdropout_converged_width_experiment\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/kdropout_converged_width_experiment\\.py:"
```
Output: 65 matches. First 3:
```
  automl_package/examples/cascade_width_experiment.py:8:`kdropout_converged_width_experiment.py`'s own reuse pattern) so every summary is directly
  automl_package/examples/moe_flexnn_comparison.py:45:    research harness in `kdropout_converged_width_experiment.py` does not need one), stated here
  automl_package/examples/capacity_ladder_results/W_CASCADE/PREREGISTRATION.md:148:  FRACTION of steps the way `kdropout_converged_width_experiment.py`'s sandwich schedule does
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/_kselection_metrics.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+_kselection_metrics\\b)|(automl_package\\.examples\\._kselection_metrics\\b)|(\\b_kselection_metrics\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/_kselection_metrics\\.py:"
```
Output: 13 matches. First 3:
```
  automl_package/examples/probreg_variational_em_step2_perinput_arbiter.py:41:import _kselection_metrics as km
  automl_package/examples/probreg_k10_sweep.py:55:from automl_package.examples._kselection_metrics import compute_kselection_metrics
  automl_package/examples/probreg_k_sweep.py:50:from automl_package.examples._kselection_metrics import compute_kselection_metrics
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/model_comparison.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+model_comparison\\b)|(automl_package\\.examples\\.model_comparison\\b)|(\\bmodel_comparison\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/model_comparison\\.py:"
```
Output: 12 matches. First 3:
```
  docs/plans/capacity_programme/probreg-report.md:79:  FIRST read `automl_package/examples/model_comparison.py`,
  docs/implementation_plan.md:151:Create a new script: `automl_package/examples/model_comparison.py`
  docs/implementation_plan.md:1779:| `examples/model_comparison.py` | **NEW** | — |  <!-- citecheck-ignore: verbatim-quoted grep hit; the shorthand is implementation_plan.md's own, not a citation made here -->
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/moe_flexnn_comparison.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+moe_flexnn_comparison\\b)|(automl_package\\.examples\\.moe_flexnn_comparison\\b)|(\\bmoe_flexnn_comparison\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/moe_flexnn_comparison\\.py:"
```
Output: 20 matches. First 3:
```
  docs/plans/capacity_programme/flexnn-moe.md:114:- Create: `automl_package/examples/moe_flexnn_comparison.py` (driver; reuses bars/metrics per
  docs/plans/capacity_programme/flexnn-core.md:96:- Create: `automl_package/examples/moe_flexnn_comparison.py` (driver) · Create (by runs):
  docs/plans/capacity_programme/flexnn-core.md:99:**STATUS 2026-07-20: NOT RUN.** The driver `automl_package/examples/moe_flexnn_comparison.py`
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/moe_regression.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+moe_regression\\b)|(automl_package\\.examples\\.moe_regression\\b)|(\\bmoe_regression\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/moe_regression\\.py:"
```
Output: 14 matches. First 3:
```
  automl_package/examples/capacity_accounting.py:25:  D2) and **MoE** (planned `moe_regression.py`, `docs/plans/capacity_programme/flexnn-moe.md`
  automl_package/examples/moe_flexnn_comparison.py:12:    elements, the toy that needed `moe_regression.py`'s new `task=ce` flag.
  automl_package/examples/moe_flexnn_comparison.py:79:import moe_regression as moe  # noqa: E402
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/noise_robustness_benchmark.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+noise_robustness_benchmark\\b)|(automl_package\\.examples\\.noise_robustness_benchmark\\b)|(\\bnoise_robustness_benchmark\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/noise_robustness_benchmark\\.py:"
```
Output: 7 matches. First 3:
```
  automl_package/examples/REPORT_phase9.md:39:`noise_robustness_benchmark.py` was extended to also evaluate
  automl_package/examples/report_a_benchmark.py:7:builders, MSE/NLL evaluate pattern), `automl_package/examples/noise_robustness_benchmark.py`
  automl_package/examples/report_a_benchmark.py:181:# Model builders — adapted from model_comparison.py / noise_robustness_benchmark.py
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/phase4_comparison.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+phase4_comparison\\b)|(automl_package\\.examples\\.phase4_comparison\\b)|(\\bphase4_comparison\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/phase4_comparison\\.py:"
```
Output: 4 matches. First 3:
```
  docs/plans/capacity_programme/flexnn-package.md:590:`automl_package/examples/phase4_comparison.py:164`, `:168`; `tests/test_phase4_regression.py:254`,
  docs/plans/capacity_programme/flexnn-package.md:646:#       automl_package/examples/phase4_comparison.py:159,164,167,168  (159/167 "soft", 164/168 "hard")
  docs/plans/capacity_programme/shared/reviews_2026-07-20/ready_package.md:202:  `phase4_comparison.py:159,164,167,168` and `moe_flexnn_comparison.py:413-414`. The write set covers
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/probabilistic_regression_showcase.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probabilistic_regression_showcase\\b)|(automl_package\\.examples\\.probabilistic_regression_showcase\\b)|(\\bprobabilistic_regression_showcase\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probabilistic_regression_showcase\\.py:"
```
Output: 5 matches. First 3:
```
  docs/architecture_analysis.md:29:| 11 | Showcase indexes 1D array as 2D | MODERATE | `probabilistic_regression_showcase.py:253-257` | Showcase crash (`IndexError`) |
  docs/architecture_analysis.md:189:**File:** `probabilistic_regression_showcase.py` L253-257
  docs/architecture_analysis.md:411:### 4.1 `probabilistic_regression_showcase.py` — INSUFFICIENT
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/probreg_ablation.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_ablation\\b)|(automl_package\\.examples\\.probreg_ablation\\b)|(\\bprobreg_ablation\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_ablation\\.py:"
```
Output: 1 matches. First 3:
```
  automl_package/examples/run_all_sweeps.sh:98:run_sweep "probreg_ablation"  "automl_package.examples.probreg_ablation"
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/probreg_elbo_prior_check.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_elbo_prior_check\\b)|(automl_package\\.examples\\.probreg_elbo_prior_check\\b)|(\\bprobreg_elbo_prior_check\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_elbo_prior_check\\.py:"
```
Output: 1 matches. First 3:
```
  docs/reports/probreg_kselection/historical/probreg_kselection_findings.md:357:- `automl_package/examples/probreg_elbo_prior_check.py`
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/probreg_identifiability_sweep.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_identifiability_sweep\\b)|(automl_package\\.examples\\.probreg_identifiability_sweep\\b)|(\\bprobreg_identifiability_sweep\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_identifiability_sweep\\.py:"
```
Output: 5 matches. First 3:
```
  automl_package/examples/run_all_sweeps.sh:95:run_sweep "identifiability"   "automl_package.examples.probreg_identifiability_sweep"
  docs/phase2_handover.md:113:`automl_package/examples/probreg_identifiability_sweep.py`:
  docs/probreg_identifiability_implementation_plan.md:411:**File:** `automl_package/examples/probreg_identifiability_sweep.py`
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/probreg_k10_sweep.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_k10_sweep\\b)|(automl_package\\.examples\\.probreg_k10_sweep\\b)|(\\bprobreg_k10_sweep\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_k10_sweep\\.py:"
```
Output: 1 matches. First 3:
```
  automl_package/examples/run_probreg_k10_sweep_safe.sh:31:        "${PY}" -m automl_package.examples.probreg_k10_sweep \
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/probreg_kselection_comparison.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_kselection_comparison\\b)|(automl_package\\.examples\\.probreg_kselection_comparison\\b)|(\\bprobreg_kselection_comparison\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_kselection_comparison\\.py:"
```
Output: 4 matches. First 3:
```
  automl_package/examples/probreg_kselection_prior_ablation.py:46:import probreg_kselection_comparison as cmp  # oracle_nll, _given, fit_mdn, mdn_nll, PROBES, MAKE, constants
  docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md:313:      `probreg_kselection_comparison.py`); best single k by held-out NLL.
  docs/plans/capacity_ladder_2026-07-09/handover/REVIEW_HANDOVER_2026-07-03.md:86:3. **The comparison harness** — `probreg_kselection_comparison.py` (real
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/probreg_kselection_diagnostic.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_kselection_diagnostic\\b)|(automl_package\\.examples\\.probreg_kselection_diagnostic\\b)|(\\bprobreg_kselection_diagnostic\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_kselection_diagnostic\\.py:"
```
Output: 1 matches. First 3:
```
  docs/reports/probreg_kselection/historical/probreg_kselection_findings.md:355:- `automl_package/examples/probreg_kselection_diagnostic.py`
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/probreg_kselection_experiments.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_kselection_experiments\\b)|(automl_package\\.examples\\.probreg_kselection_experiments\\b)|(\\bprobreg_kselection_experiments\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_kselection_experiments\\.py:"
```
Output: 1 matches. First 3:
```
  docs/reports/probreg_kselection/historical/probreg_kselection_findings.md:356:- `automl_package/examples/probreg_kselection_experiments.py`
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/probreg_kselection_prior_ablation.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_kselection_prior_ablation\\b)|(automl_package\\.examples\\.probreg_kselection_prior_ablation\\b)|(\\bprobreg_kselection_prior_ablation\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_kselection_prior_ablation\\.py:"
```
Output: 2 matches. First 3:
```
  docs/plans/capacity_ladder_2026-07-09/handover/REVIEW_HANDOVER_2026-07-03.md:82:2. **The prior ablation** — `probreg_kselection_prior_ablation.py` + results
  docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md:181:| Prior ablation (prior adds nothing under adaptive heads) | `probreg_kselection_prior_ablation.py` → `prior_ablation_results/` | run of record b0stdka43 |
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/probreg_k_sweep.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_k_sweep\\b)|(automl_package\\.examples\\.probreg_k_sweep\\b)|(\\bprobreg_k_sweep\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_k_sweep\\.py:"
```
Output: 5 matches. First 3:
```
  automl_package/examples/probreg_k10_sweep.py:1:"""ProbReg k_max=10 focused sweep — follow-up to probreg_k_sweep.py.
  automl_package/examples/classreg_k_sweep.py:5:CSV schema matches probreg_k_sweep.py for cross-join in P2.4.
  automl_package/examples/run_probreg_k_sweep_safe.sh:44:        "${PY}" -m automl_package.examples.probreg_k_sweep \
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/probreg_variational_em_step1.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_variational_em_step1\\b)|(automl_package\\.examples\\.probreg_variational_em_step1\\b)|(\\bprobreg_variational_em_step1\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_variational_em_step1\\.py:"
```
Output: 1 matches. First 3:
```
  automl_package/examples/probreg_variational_em_step2_perinput_arbiter.py:3:Step-1 (``probreg_variational_em_step1.py``) judged the WHOLE dataset at once: it asked
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/probreg_variational_em_step2_perinput_arbiter.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_variational_em_step2_perinput_arbiter\\b)|(automl_package\\.examples\\.probreg_variational_em_step2_perinput_arbiter\\b)|(\\bprobreg_variational_em_step2_perinput_arbiter\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_variational_em_step2_perinput_arbiter\\.py:"
```
Output: 6 matches. First 3:
```
  automl_package/examples/probreg_variational_em_step3_perinput_model.py:37:import probreg_variational_em_step2_perinput_arbiter as p2
  automl_package/examples/probreg_variational_em_toy_e_hump.py:39:import probreg_variational_em_step2_perinput_arbiter as p2
  automl_package/examples/capacity_ladder_x4.py:18:`probreg_variational_em_step2_perinput_arbiter.py` (independent plain-Gaussian baseline +
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/probreg_variational_em_step3_perinput_model.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_variational_em_step3_perinput_model\\b)|(automl_package\\.examples\\.probreg_variational_em_step3_perinput_model\\b)|(\\bprobreg_variational_em_step3_perinput_model\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_variational_em_step3_perinput_model\\.py:"
```
Output: 6 matches. First 3:
```
  automl_package/examples/probreg_variational_em_toy_e_hump.py:3:Step-3 (``probreg_variational_em_step3_perinput_model.py``) showed the per-input count rising
  automl_package/examples/probreg_variational_em_toy_e_hump.py:40:import probreg_variational_em_step3_perinput_model as p3
  automl_package/examples/probreg_kselection_prior_ablation.py:48:import probreg_variational_em_step3_perinput_model as p3  # effective_count, bucket_nll_per_point
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/probreg_variational_em_toy_e_hump.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+probreg_variational_em_toy_e_hump\\b)|(automl_package\\.examples\\.probreg_variational_em_toy_e_hump\\b)|(\\bprobreg_variational_em_toy_e_hump\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/probreg_variational_em_toy_e_hump\\.py:"
```
Output: 5 matches. First 3:
```
  automl_package/examples/capacity_ladder_x4.py:17:machinery from `probreg_variational_em_toy_e_hump.py::run_condition`, which itself drives
  automl_package/examples/capacity_ladder_x4.py:69:import probreg_variational_em_toy_e_hump as hump  # noqa: E402
  automl_package/examples/capacity_ladder_x4.py:80:# The arbiter's own instrument config, reused UNCHANGED from probreg_variational_em_toy_e_hump.py.
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/report_a_benchmark.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+report_a_benchmark\\b)|(automl_package\\.examples\\.report_a_benchmark\\b)|(\\breport_a_benchmark\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/report_a_benchmark\\.py:"
```
Output: 15 matches. First 3:
```
  docs/plans/capacity_programme/probreg-report.md:78:- Create: `automl_package/examples/report_a_benchmark.py` — ONLY if extension is infeasible:
  docs/plans/capacity_programme/probreg.md:40:(`automl_package/examples/report_a_benchmark.py:185-191`). The plan was wrong; the code was right.
  docs/plans/capacity_programme/probreg.md:55:| **M3** | **the sweep selector** | ONE k for the dataset, by training a **separate ORDINARY model per k** (no k-dropout, `NClassesSelectionMethod.NONE`) and scoring each on held-out data | **expensive — the reference** | generalise `select_k_for_toy` (`automl_package/examples/report_a_benchmark.py:331`), which already builds fixed-k models via `_probreg_fixed` (`:185-191`) |
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/sep_heads_vs_single_final.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+sep_heads_vs_single_final\\b)|(automl_package\\.examples\\.sep_heads_vs_single_final\\b)|(\\bsep_heads_vs_single_final\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/sep_heads_vs_single_final\\.py:"
```
Output: 1 matches. First 3:
```
  automl_package/examples/capacity_accounting.py:9:automl_package/ --include="*.py"`): the only hit is `sep_heads_vs_single_final.py:47`
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/_toy_datasets.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+_toy_datasets\\b)|(automl_package\\.examples\\._toy_datasets\\b)|(\\b_toy_datasets\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/_toy_datasets\\.py:"
```
Output: 48 matches. First 3:
```
  automl_package/examples/probreg_kselection_experiments.py:44:from automl_package.examples._toy_datasets import make_toy_a, make_toy_b
  automl_package/examples/capacity_ladder_variance_v0.py:48:import _toy_datasets as td
  automl_package/examples/probreg_variational_em_step3_perinput_model.py:35:import _toy_datasets as td
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/_variational_em.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+_variational_em\\b)|(automl_package\\.examples\\._variational_em\\b)|(\\b_variational_em\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/_variational_em\\.py:"
```
Output: 8 matches. First 3:
```
  automl_package/examples/probreg_variational_em_step2_perinput_arbiter.py:43:import _variational_em as vem
  automl_package/examples/probreg_variational_em_step1.py:36:import _variational_em as vem  # noqa: E402
  automl_package/examples/_capacity_ladder_nested.py:32:import _variational_em as vem  # _MLP, gaussian_log_density
```
Protected (PROTECTED.tsv): no
Disposition: **live**

#### `automl_package/examples/_variational_em_perinput.py`

```
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+_variational_em_perinput\\b)|(automl_package\\.examples\\._variational_em_perinput\\b)|(\\b_variational_em_perinput\\.py\\b)" --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' automl_package/ docs/ tests/ | grep -v "^automl_package/examples/_variational_em_perinput\\.py:"
```
Output: 15 matches. First 3:
```
  automl_package/examples/probreg_variational_em_step3_perinput_model.py:8:see ``_variational_em_perinput.py``) — and overlay, per input x:
  automl_package/examples/probreg_variational_em_step3_perinput_model.py:36:import _variational_em_perinput as vemp
  automl_package/examples/probreg_variational_em_toy_e_hump.py:38:import _variational_em_perinput as vemp
```
Protected (PROTECTED.tsv): no
Disposition: **live**

## Part (ii) — the bootstrap / standard-error helper duplication

**The task brief's own instruction: "RE-DERIVE THIS LIST AND THIS COUNT — do not trust the number
written here" (claimed count: fifteen).** Re-derived from scratch below. **The re-derived count is
12, not 15.**

### Re-derivation method

1. Searched for the exact seven names the brief listed, as `def` sites:
   ```bash
   grep -rn "^def _boot_se\|^def _plain_boot_se\|^def _two_sample_boot_se\|^def _paired_bootstrap_se\|^def _paired_point_bootstrap_se\|^def paired_bootstrap_se\|^def _bootstrap_col_means" automl_package/examples/*.py
   ```
   → **12 hits**, not 15 (listed in the table below).
2. Broadened past those exact names in case a renamed variant exists: `grep -rn "^def " automl_package/examples/*.py | grep -iE "boot|_se\(|standard_error|stderr"` → the same 12, plus two more candidates that turned out NOT to be bootstrap-SE primitives (`gold_mid_with_se`, `_paired_bootstrap_check` — see "Adjacent, not counted" below).
3. Cast the widest net possible — name-agnostic — by grepping for the resampling idiom itself
   (`rng.integers(0` co-occurring with a bootstrap-shaped body) across every file in
   `automl_package/examples/`, not just the `capacity_ladder_*`/`sinc_width_experiment` family the
   brief's examples all come from. This turned up 3 more files using `rng.integers(0, ...)`
   (`depth_composition_toy.py`, `joint_capacity_toy.py`, `_toy_datasets.py`) — read each hit: all
   three are unrelated data-generation code (mixture-component assignment, pool-index sampling,
   random word-sequence generation for the group-word-problem toy), not bootstrap SE. Confirmed by
   reading the surrounding code, not by the name alone.
4. **Confirmed zero hits in `automl_package/models/` or `automl_package/utils/`** — matches the
   brief's claim; the package ships no bootstrap/SE helper of its own.

The claimed count of fifteen does not match anything I could re-derive by any of the three search
strategies above; I cannot account for the extra three beyond what's listed here. This note records
what is actually on disk today (2026-07-20), not the prior claim.

### The 12, grouped by shape (for FP-9)

**Shape 1 — plain: SE of a 1-D vector's mean.** All 7 sites are thin wrappers that reuse the SAME
shared low-level resampling primitive, `_capacity_ladder._bootstrap_col_means` (defined once, at
`automl_package/examples/_capacity_ladder.py:196`, signature
`_bootstrap_col_means(s: np.ndarray, n_boot: int, block: np.ndarray | None, rng: np.random.Generator) -> np.ndarray`,
a paired/block bootstrap over column means) — none of the 7 reimplements the resampling loop itself.

| # | Site | Signature |
|---|---|---|
| 1 | `capacity_ladder_x1.py:162` `_boot_se` | `_boot_se(vec: np.ndarray, seed: int, n_boot: int = 1000) -> float` |
| 2 | `capacity_ladder_t1.py:170` `_boot_se` | `_boot_se(vec: np.ndarray, seed: int, n_boot: int = _BOOT_N) -> float` |
| 3 | `capacity_ladder_t3.py:85` `_boot_se` | `_boot_se(vec: np.ndarray, seed: int, n_boot: int = _BOOT_N) -> float` |
| 4 | `capacity_ladder_f2.py:206` `_plain_boot_se` | `_plain_boot_se(vec: np.ndarray, n_boot: int = _BOOT_N, seed: int = _BOOT_SEED) -> float` |
| 5 | `capacity_ladder_f3.py:90` `_plain_boot_se` | `_plain_boot_se(vec: np.ndarray, n_boot: int = _BOOT_N, seed: int = 0) -> float` |
| 6 | `capacity_ladder_k1k2k3.py:124` `_plain_boot_se` | `_plain_boot_se(vec: np.ndarray, n_boot: int = _BOOT_N, seed: int = 0) -> float` |
| 7 | `sinc_width_experiment.py:181` `_plain_boot_se` | `_plain_boot_se(vec: np.ndarray, n_boot: int = _BOOT_N, seed: int = _BOOT_SEED) -> float` |

All 7 bodies are the identical 3-line pattern (verified by reading each): construct `rng`, call
`cl._bootstrap_col_means(vec.reshape(-1, 1), n_boot, None, rng)`, return
`float(boot[:, 0].std(ddof=1))`. The only real variation across sites is the default `n_boot`/`seed`
values and the docstring wording — this is the strongest and safest FP-9 consolidation target: a
single `plain_boot_se(vec, n_boot, seed)` wrapping the already-shared `_bootstrap_col_means` would be
a pure rename with zero behavioral risk.

**Shape 2 — paired: SE of a paired-difference vector's mean.** Unlike shape 1, these 3 sites do **not**
call the shared `_bootstrap_col_means` — each independently reimplements the same resampling loop
inline.

| # | Site | Signature |
|---|---|---|
| 8 | `capacity_ladder_s1.py:258` `_paired_bootstrap_se` | `_paired_bootstrap_se(diffs: np.ndarray, n_boot: int = BOOT_N, seed: int = 0) -> float` |
| 9 | `capacity_ladder_k4.py:76` `paired_bootstrap_se` | `paired_bootstrap_se(diff: np.ndarray, n_boot: int, seed: int) -> float` |
| 10 | `capacity_ladder_t2.py:337` `_paired_point_bootstrap_se` | `_paired_point_bootstrap_se(diff: np.ndarray, n_boot: int, seed: int) -> float` |

All 3 bodies are the identical pattern (verified): `rng = np.random.default_rng(seed)`,
`idx = rng.integers(0, n, size=(n_boot, n))`, `boot_means = diff[idx].mean(axis=1)`,
`return float(boot_means.std(ddof=1))`. This is mathematically the same operation as shape 1 applied
to a difference vector — `_bootstrap_col_means(diff.reshape(-1,1), n_boot, None, rng)[:, 0].std(ddof=1)`
would produce the same result — but none of these 3 sites calls it; each reinvented the same 4-line
loop under a different name (`_paired_bootstrap_se` / `paired_bootstrap_se` /
`_paired_point_bootstrap_se`). This is the clearest actual duplication of the three shapes: not just
same-name-different-file, but same-logic-different-name, three times.

**Shape 3 — two-sample: SE of `mean(a) - mean(b)` for two independent (unpaired) samples.** Only one
site found across the entire sweep (not "at least five different names" for this shape — there is
exactly one implementation of this shape in the codebase):

| # | Site | Signature |
|---|---|---|
| 11 | `sinc_width_experiment.py:188` `_two_sample_boot_se` | `_two_sample_boot_se(a: np.ndarray, b: np.ndarray, n_boot: int = _BOOT_N, seed: int = _BOOT_SEED) -> float` |

Body (verified, not reused elsewhere): explicit Python `for i in range(n_boot)` loop, each iteration
independently resampling `a` and `b` and taking the difference of means, `.std(ddof=1)` over the
`n_boot` differences. Not vectorized (unlike shapes 1/2's single `rng.integers(..., size=(n_boot, n))`
call) — worth flagging for FP-9 since a vectorized two-sample version is straightforward
(`a[rng.integers(0, n_a, (n_boot, n_a))].mean(axis=1) - b[rng.integers(0, n_b, (n_boot, n_b))].mean(axis=1)`)
and would be both faster and consistent with the other two shapes' style.

**The shared primitive underlying shape 1** (counted separately, not one of the 12 "helper" duplicates
above — it's the thing being reused, not a duplicate of itself):

| Site | Signature |
|---|---|
| `_capacity_ladder.py:196` `_bootstrap_col_means` | `_bootstrap_col_means(s: np.ndarray, n_boot: int, block: np.ndarray | None, rng: np.random.Generator) -> np.ndarray` |

Returns the full `(n_boot, C)` bootstrap distribution of per-column means (plain i.i.d. row bootstrap
when `block=None`; block bootstrap over unique `block` values otherwise) — shape 1's 7 wrappers all
take `.std(ddof=1)` of column 0 of this. This is already `import _capacity_ladder as cl`'d by name
across the shape-1 sites (verified: `cl._bootstrap_col_means(...)` appears in all 7 bodies), so it is
NOT itself a zero-caller candidate: `_capacity_ladder` is one of the 72 `live` rows in Part (i) above
(33 matches under the FP-7.a search form, not in PROTECTED.tsv).

### Adjacent — same problem, NOT counted in the 12 (read and excluded, not just skipped by name)

- **`capacity_ladder_x4.py:128` `_tercile_means_se`** —
  `_tercile_means_se(x: np.ndarray, v: np.ndarray) -> dict[str, dict[str, float]]`. Computes a
  standard error, but **analytically** (`vv.std(ddof=1) / np.sqrt(vv.size)`), not by bootstrap
  resampling at all. Same conceptual purpose (an SE-of-a-mean helper) as shape 1, genuinely different
  method. Flagging for FP-9's attention as a fourth, non-bootstrap variant of "give me an SE" that
  exists in the same family of scripts — not merged into the 12 because it is not a
  bootstrap-resampling duplicate, it is a different statistical technique.
- **`capacity_ladder_t3.py:96` `gold_mid_with_se`** — a domain-specific consumer (tercile-mean-plus-SE
  for a specific gold curve), not a primitive; internally calls this same file's `_boot_se` (shape 1,
  site 3 above). Not counted separately.
- **`capacity_ladder_f2.py:212` `_paired_bootstrap_check`** — a domain-specific consumer (threshold
  check `|mean_delta| < 2*SE`), not a primitive; internally calls this same file's `_plain_boot_se`
  (shape 1, site 4 above). Not counted separately.
- **`capacity_ladder_k6.py`/`capacity_ladder_s2.py`/`capacity_ladder_t2.py`'s calls to `cl.knee(...,
  n_boot=...)`** — `_capacity_ladder.knee` is a separate, already-shared function (not duplicated
  across files; called via `cl.knee`) that uses bootstrap SE internally for changepoint/knee
  detection. Not a duplicate helper — it is exactly the "already centralized, call by name" pattern
  FP-9 should extend to the other three shapes, not a new thing to fix.

### Non-goals honored

Per the task's explicit non-goal: nothing above was de-duplicated or modified. No file was edited.
This section only inventories, with signatures and shapes, for FP-9 to build against.

---

## Coverage boundary — what this note did NOT sweep, and why

This section exists precisely so silence never reads as a clean bill: everything left NOT swept by
this task is named explicitly below, with the reason.

- **Part (i)'s zero-caller sweep used exactly the mandated search scope, `automl_package/ docs/
  tests/`, for all 104 modules — no example script was skipped.** But that scope is, by the task's own
  specification, narrower than the whole repository: it does **not** include repo-root files
  (`README.md`, `CLAUDE.md`, `RESUME.md`, `CHANGELOG.md`, `ARCHIVE.md`, `SESSION_JOURNAL.md`,
  `SESSION_JOURNAL_2.md`, `install.sh`, `requirements.txt`, `pyproject.toml`, `ruff.toml`) and does
  not include anything outside this repository (external notebooks, other repos in the `~/dev/`
  workspace, W&B run configs, etc.). I ran a **supplementary, non-mandated** check of the root-level
  doc set above against all 21 dead-candidates and found 8 real citations (see the caveat table in
  Part (i)) — this is reported as an addition on top of the required sweep, not a substitute for it,
  and I did **not** extend the supplementary check to the protected/live modules (104 - 21 = 83
  modules) or to anything outside the repo. A later reader should not assume "not swept" here means
  "confirmed absent" anywhere outside `automl_package/ docs/ tests/` plus the specific 11-file
  root-doc set checked for the 21 dead-candidates only.
- **Part (i) did not check for callers via string-based dynamic import** (`importlib.import_module`,
  `exec`, `__import__`, `getattr(module, name)` dispatch tables). Grepped
  `grep -rln "importlib\|__import__" automl_package/examples/*.py automl_package/*.py` before writing
  this note — result: no hits in `automl_package/examples/` or at the package top level (isolated
  hits, if any, were not chased further given the FP-7.a search form is what the task mandates and
  what FP-8's gate checks against). If any script builds a module name from a string and imports it
  dynamically, that caller would be invisible to this sweep's static grep, dynamic-import or not — I
  did not verify absence of this pattern exhaustively across every file, only ran the one grep above.
- **Part (i) did not open any file to check whether a module is imported inside a docstring code
  example, a commented-out block, or a `# noqa` guarded conditional import** — the grep matches text,
  not parsed/executed Python; a module could show as "live" from a comment that doesn't actually run,
  or (more relevant to safety) could show as `dead-candidate` while still being reachable through a
  form the regex doesn't match (e.g. `importlib.import_module(f"{prefix}_experiment")`, string
  concatenation, or a shell wildcard glob in a launcher script that never spells out the filename).
  Not verified beyond the one importlib/`__import__` grep above.
- **Part (ii)'s bootstrap/SE re-derivation searched only `automl_package/examples/*.py`, per the
  brief's own stated scope** ("zero exist in `automl_package/models/` or `automl_package/utils/`" —
  reverified true, see step 4 of the re-derivation method). It did not search `tests/` for bootstrap/SE
  test helpers, which could be a further duplication surface FP-9 should check when it builds the
  unified helper (not chased here — out of this task's stated scope, which is examples/ drivers).
- **Neither part re-ran any experiment, touched any results directory, or re-verified any certified
  claim.** This is a pure static-text inventory.

## Forward reference to park with the root

None. Every path this note cites (`docs/plans/capacity_programme/shared/PROTECTED.tsv`,
`docs/plans/capacity_programme/shared/reviews_2026-07-20/dup_inventory.md`,
`automl_package/examples/_capacity_ladder.py`, and the individual example-script paths throughout)
resolves on disk as of this run.
