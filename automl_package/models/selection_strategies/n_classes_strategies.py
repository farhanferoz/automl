"""Thin re-export shim — the n_classes selection strategies now live in `automl_package.models.flexnn.strategies.n_classes`.

Kept here (rather than deleted) so every existing `from automl_package.models.
selection_strategies.n_classes_strategies import ...` call site keeps resolving unchanged
(2026-07-21, `docs/plans/capacity_programme/flexnn-package.md` Task FP-11 — the FlexNN family
gets one home, MASTER Decision 19's boundary rule; the `automl_package/examples/convergence.py`
precedent: move the logic, leave the shim, do not rewrite callers). `probabilistic_regression_net.py`
is the live external consumer and is untouched — it keeps importing from this old path.
"""

from __future__ import annotations

from automl_package.models.flexnn.strategies.n_classes import (
    GumbelSoftmaxStrategy,
    NestedStrategy,
    NoneStrategy,
    ReinforceStrategy,
    SoftGatingStrategy,
    SteStrategy,
)

__all__ = [
    "GumbelSoftmaxStrategy",
    "NestedStrategy",
    "NoneStrategy",
    "ReinforceStrategy",
    "SoftGatingStrategy",
    "SteStrategy",
]
