"""Global (whole-dataset) capacity selection primitives (capacity-programme Task FP-9.a).

One capacity for the WHOLE dataset, not one per input -- see MASTER Decision 18
(`docs/plans/capacity_programme/MASTER.md`) for why the two selection rules in this programme are
deliberately different and both retained. `cheapest_within_tolerance` below answers "which single
capacity for *this dataset*", selecting the smallest (cheapest) capacity in a held-out error curve
whose error is not meaningfully worse than the best capacity's, where "meaningfully worse" is
calibrated from the data itself: the paired difference against the best capacity must exceed TWICE
a bootstrap-estimated standard error of that difference. Rule of record:
`docs/reports/probreg_kselection/probreg_kselection.md` §3.2 -- "require a difference of at least
twice an estimated standard error of that noise ... before calling it real", the noise estimated by
bootstrap resampling. Imported here, not re-derived.

**This is NOT `automl_package.models.common.distilled_router._cheapest_within_tolerance_labels`,
and it must not be built by generalising that function, and it does not replace it.** That function
is a PER-ROW labeller for the per-input distilled router: it takes an `(n_samples, n_capacities)`
error TABLE and labels EVERY ROW independently with a fixed relative margin
(`error <= (1 + 0.25) * row_min`). `cheapest_within_tolerance` below answers a different question --
"which ONE capacity for the whole dataset" -- from a different input shape (a held-out error curve
with one column per capacity, evaluated once for the whole dataset, not per-row), with a
noise-calibrated (bootstrap SE) margin instead of a hand-set relative one. Different input
semantics, different tolerance rule, different consumer (the global arms -- W-SHARED/W-SWEEP,
ProbReg M1/M3, the depth equivalents -- never the distilled router). MASTER Decision 18: **both
exist; neither replaces the other**, because a per-input decision has one row's worth of evidence
and no standard error is estimable from a single observation, while a global chooser reads a whole
held-out curve, over which a bootstrap standard error is exactly the right notion of noise.
"""

from __future__ import annotations

import numpy as np

from automl_package.utils.numerics import bootstrap_se

DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 0
TOLERANCE_SE_MULTIPLE = 2.0  # "twice an estimated standard error" -- probreg_kselection.md §3.2
_ERROR_TABLE_NDIM = 2  # (n_samples, n_capacities)


def cheapest_within_tolerance(error_table: np.ndarray, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> int:
    """Smallest-index capacity whose held-out error is within 2 bootstrap SE of the best capacity's.

    For each capacity, in cheapest-first order, computes the paired per-example difference against
    the best (lowest column-mean) capacity and its bootstrap standard error; returns the first
    capacity whose mean difference does not exceed `TOLERANCE_SE_MULTIPLE` times that SE. The best
    capacity itself always qualifies (its own diff column is all-zero), so the loop always
    terminates on or before reaching it.

    Args:
        error_table: `(n_samples, n_capacities)` held-out per-sample error at each capacity, lower
            is better. Capacity columns are ordered CHEAPEST FIRST (index 0 = cheapest), matching
            `distilled_router`'s `capacity_grid` convention.
        n_boot: bootstrap resamples for the paired-difference standard error.
        seed: RNG seed for the bootstrap resample -- determinism (FP-9.b).

    Returns:
        0-based column index of the smallest (cheapest) capacity that is not meaningfully worse
        than the best capacity.

    Raises:
        ValueError: `error_table` is not 2-D, or has zero capacity columns.
    """
    table = np.asarray(error_table, dtype=np.float64)
    if table.ndim != _ERROR_TABLE_NDIM:
        raise ValueError(f"error_table must be 2-D (n_samples, n_capacities), got shape {table.shape}")
    n_capacities = table.shape[1]
    if n_capacities == 0:
        raise ValueError("error_table must have at least one capacity column.")

    column_means = table.mean(axis=0)
    best_idx = int(np.argmin(column_means))
    best_col = table[:, best_idx]

    for capacity_idx in range(n_capacities):
        diff = table[:, capacity_idx] - best_col
        mean_diff = float(diff.mean())
        se_diff = bootstrap_se(diff, n_boot=n_boot, seed=seed)
        if mean_diff <= TOLERANCE_SE_MULTIPLE * se_diff:
            return capacity_idx
    return best_idx  # unreachable: best_idx's own diff column is all-zero, so it always qualifies
