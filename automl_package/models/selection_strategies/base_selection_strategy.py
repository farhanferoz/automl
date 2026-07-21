"""Thin re-export shim — `BaseSelectionStrategy` now lives in `automl_package.models.flexnn.strategies.base`.

Kept here (rather than deleted) so every existing `from automl_package.models.
selection_strategies.base_selection_strategy import ...` call site keeps resolving unchanged
(2026-07-21, `docs/plans/capacity_programme/flexnn-package.md` Task FP-11 — the FlexNN family
gets one home, MASTER Decision 19's boundary rule; the `automl_package/examples/convergence.py`
precedent: move the logic, leave the shim, do not rewrite callers).
"""

from __future__ import annotations

from automl_package.models.flexnn.strategies.base import DIRECT_REGRESSION_K_SENTINEL, BaseSelectionStrategy

__all__ = ["DIRECT_REGRESSION_K_SENTINEL", "BaseSelectionStrategy"]
