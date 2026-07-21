"""Thin re-export shim — the independent-weights selection strategies now live in `automl_package.models.flexnn.strategies.independent_weights`.

Kept here (rather than deleted) so every existing `from automl_package.models.
selection_strategies.independent_weights_strategies import ...` call site keeps resolving
unchanged (2026-07-21, `docs/plans/capacity_programme/flexnn-package.md` Task FP-11 — the FlexNN
family gets one home, MASTER Decision 19's boundary rule; the `automl_package/examples/
convergence.py` precedent: move the logic, leave the shim, do not rewrite callers).
"""

from __future__ import annotations

from automl_package.models.flexnn.strategies.independent_weights import (
    IndependentWeightsGumbelSoftmaxStrategy,
    IndependentWeightsNestedStrategy,
    IndependentWeightsNoneStrategy,
    IndependentWeightsReinforceStrategy,
    IndependentWeightsSoftGatingStrategy,
    IndependentWeightsSteStrategy,
)

__all__ = [
    "IndependentWeightsGumbelSoftmaxStrategy",
    "IndependentWeightsNestedStrategy",
    "IndependentWeightsNoneStrategy",
    "IndependentWeightsReinforceStrategy",
    "IndependentWeightsSoftGatingStrategy",
    "IndependentWeightsSteStrategy",
]
