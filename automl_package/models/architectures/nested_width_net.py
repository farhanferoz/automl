"""Thin re-export shim — the nested-width architectures now live in `automl_package.models.flexnn.width.architectures`.

Kept here (rather than deleted) so every existing `from automl_package.models.architectures.
nested_width_net import ...` call site keeps resolving unchanged (2026-07-21,
`docs/plans/capacity_programme/flexnn-package.md` Task FP-11 — the FlexNN family gets one home,
MASTER Decision 19's boundary rule; the `automl_package/examples/convergence.py` precedent: move
the logic, leave the shim, do not rewrite callers).
"""

from __future__ import annotations

from automl_package.models.flexnn.width.architectures import (
    W_MAX_DEFAULT,
    IndependentWidthNet,
    NestedWidthNet,
    SharedReadoutPerWidthAffineNet,
    SharedTrunkPerWidthHeadNet,
)

__all__ = [
    "W_MAX_DEFAULT",
    "IndependentWidthNet",
    "NestedWidthNet",
    "SharedReadoutPerWidthAffineNet",
    "SharedTrunkPerWidthHeadNet",
]
