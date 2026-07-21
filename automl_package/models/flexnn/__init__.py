"""FlexNN — the one home for the flexible-capacity model family (width, depth, routing).

Consolidated 2026-07-21 (`docs/plans/capacity_programme/flexnn-package.md` Task FP-11) from four
previously scattered locations (`models/architectures/`, `models/flexible_width_network.py`,
`models/flexible_neural_network.py`, `models/independent_weights_flexible_neural_network.py`,
`models/selection_strategies/`, `models/common/distilled_router.py`). The old paths remain as
re-export shims so existing call sites keep resolving unchanged; new code should import from here.

This top-level package re-exports the models and mechanisms that make up the family. The
per-strategy classes under `flexnn.strategies` (`layer`, `independent_weights`, `n_classes`) are
NOT re-exported here — `layer` and `n_classes` each define their own `NoneStrategy`,
`GumbelSoftmaxStrategy`, etc., so flattening them into one namespace would collide; import those
from their submodule directly, e.g. `from automl_package.models.flexnn.strategies.layer import
NestedStrategy`.

`automl_package.models.probabilistic_regression` and `automl_package.models.architectures.
probabilistic_regression_net` are NOT part of FlexNN — ProbReg is a model that USES a capacity
dial, not a capacity mechanism, and stays beside its sibling `classifier_regression.py`.
"""

from __future__ import annotations

from automl_package.models.flexnn.depth.independent_weights import IndependentWeightsFlexibleNN
from automl_package.models.flexnn.depth.model import FlexibleHiddenLayersNN
from automl_package.models.flexnn.routing import DEFAULT_TOLERANCE, DistilledCapacityRouter
from automl_package.models.flexnn.width.architectures import (
    W_MAX_DEFAULT,
    IndependentWidthNet,
    NestedWidthNet,
    SharedReadoutPerWidthAffineNet,
    SharedTrunkPerWidthHeadNet,
)
from automl_package.models.flexnn.width.model import FlexibleWidthNN

__all__ = [
    "DEFAULT_TOLERANCE",
    "W_MAX_DEFAULT",
    "DistilledCapacityRouter",
    "FlexibleHiddenLayersNN",
    "FlexibleWidthNN",
    "IndependentWeightsFlexibleNN",
    "IndependentWidthNet",
    "NestedWidthNet",
    "SharedReadoutPerWidthAffineNet",
    "SharedTrunkPerWidthHeadNet",
]
