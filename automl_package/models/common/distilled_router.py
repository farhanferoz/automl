"""Thin re-export shim — `DistilledCapacityRouter` now lives in `automl_package.models.flexnn.routing`.

Kept here (rather than deleted) so every existing `from automl_package.models.common.
distilled_router import ...` call site keeps resolving unchanged (2026-07-21, `docs/plans/
capacity_programme/flexnn-package.md` Task FP-11 — the FlexNN family gets one home, MASTER
Decision 19's boundary rule; the `automl_package/examples/convergence.py` precedent: move the
logic, leave the shim, do not rewrite callers).
"""

from __future__ import annotations

from automl_package.models.flexnn.routing import (
    DEFAULT_HIDDEN,
    DEFAULT_LR,
    DEFAULT_N_EPOCHS,
    DEFAULT_TOLERANCE,
    DistilledCapacityRouter,
    _cheapest_within_tolerance_labels,  # noqa: F401 -- private, kept resolving for tests/test_distilled_router.py and tests/test_phase3_dynamic_k.py
)

__all__ = [
    "DEFAULT_HIDDEN",
    "DEFAULT_LR",
    "DEFAULT_N_EPOCHS",
    "DEFAULT_TOLERANCE",
    "DistilledCapacityRouter",
]
