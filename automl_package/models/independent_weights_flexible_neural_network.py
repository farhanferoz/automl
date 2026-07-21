"""Thin re-export shim — `IndependentWeightsFlexibleNN` now lives in `automl_package.models.flexnn.depth.independent_weights`.

Kept here (rather than deleted) so every existing `from automl_package.models.
independent_weights_flexible_neural_network import IndependentWeightsFlexibleNN` call site keeps
resolving unchanged (2026-07-21, `docs/plans/capacity_programme/flexnn-package.md` Task FP-11 —
the FlexNN family gets one home, MASTER Decision 19's boundary rule; the `automl_package/examples/
convergence.py` precedent: move the logic, leave the shim, do not rewrite callers).
"""

from __future__ import annotations

from automl_package.models.flexnn.depth.independent_weights import IndependentWeightsFlexibleNN

__all__ = ["IndependentWeightsFlexibleNN"]
