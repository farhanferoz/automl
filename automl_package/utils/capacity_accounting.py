"""Capacity-accounting primitives -- param/FLOP counting shared by every capacity-dial family.

Split out of `automl_package/examples/capacity_accounting.py` (2026-07-20,
`docs/plans/capacity_programme/flexnn-package.md` Task FP-1) to break a load-time circular
import: that script imports `FlexibleHiddenLayersNN`/`FlexibleWidthNN` from the package at module
level, while the package's `fit_router` methods need `executed_flops` -- previously resolved with
a deferred, inside-method import, which is exactly the pattern FP-1 exists to remove.

**This module knows about NO specific network class.** `param_count` and `executed_flops` are
`functools.singledispatch` functions; every consumer registers ITS OWN type against them from
wherever that type is defined:
- `FlexibleHiddenLayersNN.FlexibleNNModule` and `FlexibleWidthNN.FlexibleWidthNNModule` register
  themselves in their own model files (`models/flexible_neural_network.py`,
  `models/flexible_width_network.py`), right after their class definitions.
- The four `nested_width_net.py` width architectures (`NestedWidthNet`,
  `SharedTrunkPerWidthHeadNet`, `IndependentWidthNet`, `SharedReadoutPerWidthAffineNet`) register
  from `automl_package/examples/capacity_accounting.py` (the shim) -- they stay on the examples
  side until FP-2 moves the classes themselves into the package.
This module importing any of those classes to register them centrally would recreate the same
circular-import problem FP-1 removes (a diamond: model file -> this module -> the OTHER model
file -> this module). Per-type self-registration via `.register()` avoids it entirely and is the
standard `functools.singledispatch` idiom -- no bespoke registration hook needed.

The two shape descriptors below (`DepthNetShapeDescriptor`, `MoEShapeDescriptor`) have no model
class yet, so they register directly, here.

Under MASTER Decision 2 (MSE-only across the capacity strands,
`docs/plans/capacity_programme/MASTER.md`), the width nets' `logvar_head` is dead weight at
inference: `param_count` exposes `path_filter` so a caller can EXCLUDE it explicitly (e.g.
`path_filter=LOGVAR_HEAD_PATH_SUBSTRING`), while `executed_flops` excludes it UNCONDITIONALLY
(no flag) since no MSE-only deployment ever calls it.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import torch.nn as nn

LOGVAR_HEAD_PATH_SUBSTRING = "logvar"  # nested_width_net.py's NestedWidthNet/IndependentWidthNet attribute name


# ---------------------------------------------------------------------------
# Shape descriptors for families with no real class yet (see module docstring).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthNetShapeDescriptor:
    """Shape of the depth-net family's shared-blocks-plus-shared-head arm.

    Planned `NestedDepthNet` (`docs/plans/capacity_programme/depth.md` Task D2 -- not built at S2
    authoring time). One `input_size -> hidden_size` block, `max_depth - 1` chained
    `hidden_size -> hidden_size` blocks, and ONE shared `hidden_size -> output_size` readout
    applied after the routed depth's blocks (D2: "shared blocks + ONE shared output head over
    every depth prefix"). D2's other two planned arms (per-depth heads, independent nets) are not
    represented here -- this descriptor covers only the shared-head arm; extend when the real
    classes land instead of guessing their per-arm deltas.
    """

    input_size: int
    hidden_size: int
    output_size: int
    max_depth: int


@dataclass(frozen=True)
class MoEShapeDescriptor:
    """Shape of the planned MoE family.

    `MoERegressionNet` (`docs/plans/capacity_programme/flexnn-moe.md` Task M2 -- not built at S2
    authoring time). M2's frozen spec: each expert is `Linear(d_in -> expert_hidden) -> tanh ->
    Linear(expert_hidden -> 1)`; the router is a single `Linear(d_in -> n_experts)`. `n_experts`
    is the TOTAL expert count (the router's output width); `executed_flops`'s `config` argument
    is `top_k`, the number of experts actually dispatched to.
    """

    d_in: int
    expert_hidden: int
    n_experts: int


# ---------------------------------------------------------------------------
# Shared arithmetic primitives.
# ---------------------------------------------------------------------------


def _linear_macs(in_features: int, out_features: int) -> int:
    """Multiply-add (MAC) count of one `Linear(in_features, out_features)` matmul.

    This module's convention: MACs = `in_features * out_features`. Bias adds (`O(out_features)`,
    dominated by the matmul term) are NOT separately counted -- a stated convention of this module,
    not a claim about any external profiler's convention.
    """
    return int(in_features) * int(out_features)


def _linear_params(in_features: int, out_features: int) -> int:
    """Trainable parameter count of one `Linear(in_features, out_features)`: weight + bias."""
    return int(in_features) * int(out_features) + int(out_features)


def _sequential_linear_macs(seq: nn.Sequential) -> int:
    """Sums `_linear_macs` over every `nn.Linear` directly inside a `Sequential`.

    Non-`Linear` layers (activation, `BatchNorm1d`, `Dropout`) contribute no matmul and are
    skipped -- this makes the helper correct for both FlexNN's `hidden_layers_blocks[i]` (Linear +
    optional BatchNorm + activation + optional Dropout) and its `n_predictor` (alternating Linear
    + activation) without needing a separate formula for each.
    """
    return sum(_linear_macs(m.in_features, m.out_features) for m in seq if isinstance(m, nn.Linear))


# ---------------------------------------------------------------------------
# param_count -- one generic nn.Module implementation covers every registered class; the two shape
# descriptors get explicit analytic overloads.
# ---------------------------------------------------------------------------


@functools.singledispatch
def param_count(net: object, path_filter: str | None = None) -> int:
    """Trainable parameter count of `net`.

    Args:
        net: an `nn.Module` (any registered network class -- see module docstring for who
            registers where), or one of this module's shape descriptors
            (`DepthNetShapeDescriptor`, `MoEShapeDescriptor`) for a family whose real class does
            not exist yet.
        path_filter: if given, EXCLUDES any parameter whose dotted name contains this substring
            (e.g. `LOGVAR_HEAD_PATH_SUBSTRING`, to drop the width nets' unused-under-MSE logvar
            heads). No effect on the shape-descriptor overloads -- they have no named parameters
            to filter; the argument is accepted there for signature symmetry only.

    Returns:
        Total trainable (`requires_grad=True`) parameter count, honoring `path_filter` if given.

    Raises:
        TypeError: `net`'s type has no registered accounting implementation.
    """
    del path_filter  # unused in the dispatch stub -- always raises before it would be read
    raise TypeError(f"param_count: no accounting implementation registered for {type(net).__name__}")


@param_count.register(nn.Module)
def _param_count_module(net: nn.Module, path_filter: str | None = None) -> int:
    """Generic `nn.Module` implementation: sums `numel()` over trainable params, name-filtered."""
    return sum(p.numel() for name, p in net.named_parameters() if p.requires_grad and (path_filter is None or path_filter not in name))


@param_count.register(DepthNetShapeDescriptor)
def _param_count_depth_descriptor(net: DepthNetShapeDescriptor, path_filter: str | None = None) -> int:
    """Total params of the shared-blocks-plus-shared-head arm (see class docstring)."""
    del path_filter  # no named parameters on a shape descriptor
    total = _linear_params(net.input_size, net.hidden_size)  # block 0
    total += (net.max_depth - 1) * _linear_params(net.hidden_size, net.hidden_size)  # blocks 1..max_depth-1
    total += _linear_params(net.hidden_size, net.output_size)  # shared output head
    return total


@param_count.register(MoEShapeDescriptor)
def _param_count_moe_descriptor(net: MoEShapeDescriptor, path_filter: str | None = None) -> int:
    """Total params of ALL `n_experts` experts plus the router (params exist regardless of `top_k`)."""
    del path_filter  # no named parameters on a shape descriptor
    router = _linear_params(net.d_in, net.n_experts)
    per_expert = _linear_params(net.d_in, net.expert_hidden) + _linear_params(net.expert_hidden, 1)
    return router + net.n_experts * per_expert


# ---------------------------------------------------------------------------
# executed_flops -- one registration per family, each registered from wherever its class is
# defined (see module docstring for why this module does not register FlexNN/FlexibleWidthNN/the
# nested_width_net.py classes itself).
# ---------------------------------------------------------------------------


@functools.singledispatch
def executed_flops(net: object, config: int) -> int:
    """Analytic multiply-add (MAC) count of one forward pass of `net` AT ITS ROUTED CAPACITY.

    Not a timed benchmark (analytic-only by design). "Routed capacity" (`config`) means: width
    nets -> the routed width `k`; FlexNN -> the selected depth `d`; the depth-net descriptor ->
    its routed depth `d`; the MoE descriptor -> `top_k` (experts actually executed).

    Args:
        net: any registered network class (see module docstring), or one of this module's shape
            descriptors for a family with no real class yet.
        config: the routed capacity level (see above); family-specific range checks raise
            `ValueError` if out of bounds.

    Returns:
        Multiply-add count for that one forward call at the given routed capacity.

    Raises:
        TypeError: `net`'s type has no registered accounting implementation.
    """
    del config  # unused in the dispatch stub -- always raises before it would be read
    raise TypeError(f"executed_flops: no accounting implementation registered for {type(net).__name__}")


@executed_flops.register(DepthNetShapeDescriptor)
def _executed_flops_depth_descriptor(net: DepthNetShapeDescriptor, config: int) -> int:
    """Routed depth-`config` MACs for the depth-net descriptor (see class docstring).

    Same block-chain-plus-output-head shape as FlexNN, so the same formula shape applies: one
    `input_size -> hidden_size` block, `config - 1` chained `hidden_size -> hidden_size` blocks,
    one shared `hidden_size -> output_size` readout.
    """
    d = config
    if not (1 <= d <= net.max_depth):
        raise ValueError(f"config={d} out of range [1, {net.max_depth}]")
    total = _linear_macs(net.input_size, net.hidden_size)  # block 0
    total += (d - 1) * _linear_macs(net.hidden_size, net.hidden_size)  # blocks 1..d-1
    total += _linear_macs(net.hidden_size, net.output_size)  # shared output head
    return total


@executed_flops.register(MoEShapeDescriptor)
def _executed_flops_moe_descriptor(net: MoEShapeDescriptor, config: int) -> int:
    """Routed top-`config` MACs for the MoE descriptor (see class docstring).

    The router ALWAYS scores every one of `n_experts` (top-k selection needs every gate logit
    before it can pick), so router MACs are fixed regardless of `config`; only `config` experts'
    own forward passes (`Linear(d_in, expert_hidden) -> Linear(expert_hidden, 1)`, tanh excluded
    -- no matmul) run.
    """
    top_k = config
    if not (1 <= top_k <= net.n_experts):
        raise ValueError(f"config={top_k} out of range [1, {net.n_experts}]")
    router_macs = _linear_macs(net.d_in, net.n_experts)
    per_expert_macs = _linear_macs(net.d_in, net.expert_hidden) + _linear_macs(net.expert_hidden, 1)
    return router_macs + top_k * per_expert_macs


# ---------------------------------------------------------------------------
# Selection-cost accounting (capacity-programme Task FP-9.c). The rest of this module prices a
# network AT a given capacity; it had no notion of the cost of CHOOSING that capacity, which is
# what makes an unqualified "cheaper at inference" claim unfalsifiable. Analytic, matching this
# module's design (no wall-clock harness). Family-agnostic: width, depth, ProbReg-k and MoE all
# reach it through the same `executed_flops` dispatcher, with no per-family branching here.
#
# Drafted by the FP-9 worker and applied by the ROOT: `write_set_guard` blocks a second subagent
# from writing a file another already wrote this session, by design. One change was made in
# review -- `hidden` is REQUIRED, see `router_fit_cost`.
# ---------------------------------------------------------------------------

_BACKWARD_FORWARD_MAC_RATIO = 2  # backward = grad-input matmul + grad-weight matmul, each ~1x the forward MACs


def _mlp_forward_macs(in_dim: int, hidden: Sequence[int], out_dim: int) -> int:
    """Forward-pass MAC count of a plain `Linear -> activation -> ... -> Linear` MLP."""
    dims = [in_dim, *hidden, out_dim]
    return sum(_linear_macs(d_in, d_out) for d_in, d_out in itertools.pairwise(dims))


def router_fit_cost(in_dim: int, n_capacities: int, n_samples: int, n_epochs: int, hidden: Sequence[int]) -> int:
    """Analytic MAC cost of TRAINING a distilled capacity router (selection mechanism 1 of 3).

    Full-batch training: `n_epochs` epochs, each a forward and a backward pass over all
    `n_samples`. Shaped like `DistilledCapacityRouter`'s `in_dim -> hidden -> n_capacities` MLP,
    without importing it -- this module imports nothing from `models/` (see the module docstring).

    `hidden` is REQUIRED rather than defaulted to the router's own `DEFAULT_HIDDEN`: that constant
    is a shared global whose value only `flexnn-package.md` FP-5 may change, and a default copied
    here would drift out of step with it silently. Pass the router's actual configuration.

    Args:
        in_dim: Router input dimensionality.
        n_capacities: Size of the capacity grid the router selects over (its output width).
        n_samples: Held-out samples the router is fit on.
        n_epochs: Full-batch training epochs.
        hidden: The router MLP's hidden-layer sizes.

    Returns:
        Total MACs (forward plus backward) across every epoch and sample.
    """
    forward_macs = _mlp_forward_macs(in_dim, hidden, n_capacities)
    return n_epochs * n_samples * forward_macs * (1 + _BACKWARD_FORWARD_MAC_RATIO)


def held_out_read_cost(net: object, capacity_grid: Sequence[int], n_samples: int) -> int:
    """Analytic MAC cost of scoring every capacity once on a selection set (mechanism 2 of 3).

    One forward pass per sample per candidate capacity; no backward pass. Family-agnostic: `net`
    is any type registered against `executed_flops`.

    Args:
        net: A registered network or shape descriptor -- see `executed_flops`.
        capacity_grid: Candidate capacities to score.
        n_samples: Held-out samples scored at each capacity.

    Returns:
        Total MACs of scoring every capacity in `capacity_grid` once over `n_samples` examples.
    """
    return n_samples * sum(executed_flops(net, capacity) for capacity in capacity_grid)


def sweep_cost(net: object, capacity_grid: Sequence[int], n_train_samples: int, n_epochs: int) -> int:
    """Analytic MAC cost of a sweep -- training a dedicated model per capacity (mechanism 3 of 3).

    This is the expensive reference the cheap mechanisms are measured against.

    Args:
        net: A registered network or shape descriptor -- see `executed_flops`.
        capacity_grid: Capacities, each trained as its own model.
        n_train_samples: Training samples per capacity.
        n_epochs: Full-batch training epochs per capacity.

    Returns:
        Total MACs (forward plus backward) of training one model per capacity in `capacity_grid`.
    """
    return sum(n_epochs * n_train_samples * executed_flops(net, capacity) * (1 + _BACKWARD_FORWARD_MAC_RATIO) for capacity in capacity_grid)
