"""S2 — the ONE source of params/FLOPs numbers for reports (b)/(c) and the strand-4 deploy bars.

`docs/plans/capacity_programme/shared/metrics-accounting.md` Task S2. Analytic (not timed)
accounting: `param_count` (trainable parameter count) and `executed_flops` (multiply-add count
of one forward pass AT A ROUTED CAPACITY — width `k`, depth `d`, or MoE `top_k`) for the
programme's architecture families.

**Search-first result** (ladder rung 2, `grep -rn "def count_params|flops|def .*param_count"
automl_package/ --include="*.py"`): the only hit is `sep_heads_vs_single_final.py:47`
(`_param_count`), a private one-off (`sum(p.numel() for p in m.model.parameters() if
p.requires_grad)` on one specific model type, no path filtering, no FLOPs, not exported). Nothing
generalizable to extend — this module is new.

**Families and how each is accounted:**
- **Width nets** (`nested_width_net.py`: `NestedWidthNet`, `SharedTrunkPerWidthHeadNet`,
  `IndependentWidthNet`, `SharedReadoutPerWidthAffineNet`) — derived from the REAL classes'
  `nn.Linear` shapes. `param_count` is generic over any `nn.Module` (one implementation covers
  all four). `executed_flops` is registered per class because each shares weights differently
  (see each registration's docstring for its derivation) — do NOT copy one class's formula onto
  another; the S2 brief explicitly warns against a blind "3k+1" guess.
- **FlexNN** (`models/flexible_neural_network.py`: `FlexibleHiddenLayersNN.FlexibleNNModule`) —
  derived from the REAL class, mirroring `hard_forward`'s actual compute-saving code path
  (models/flexible_neural_network.py:128-159).
- **Depth nets** (planned `nested_depth_net.py`, `docs/plans/capacity_programme/depth.md` Task
  D2) and **MoE** (planned `moe_regression.py`, `docs/plans/capacity_programme/flexnn-moe.md`
  Task M2) have NO real class yet at S2 authoring time (2026-07-16) — both are still "Create"
  tasks in their strand files. Per the S2 brief's explicit fallback ("if a family's net class
  isn't importable cleanly, accept a lightweight shape descriptor ... and document it"), these
  two use `DepthNetShapeDescriptor` / `MoEShapeDescriptor` dataclasses sized from the shapes each
  strand file already specifies (D2: "shared blocks + ONE shared output head"; M2: "experts are
  small MLPs `Linear(d_in->H) -> tanh -> Linear(H->1)`; router is a linear layer over the
  input"). When the real classes land, add `nn.Module`-based registrations here alongside these
  (do not delete the descriptors — historical ledger numbers may still cite them).

Under MASTER Decision 2 (MSE-only across the capacity strands,
`docs/plans/capacity_programme/MASTER.md`), the width nets' `logvar_head` is dead weight at
inference: `param_count` exposes `path_filter` so a caller can EXCLUDE it explicitly (e.g.
`path_filter=LOGVAR_HEAD_PATH_SUBSTRING`), while `executed_flops` excludes it UNCONDITIONALLY
(no flag) since no MSE-only deployment ever calls it.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_accounting.py --selftest
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from dataclasses import dataclass

import torch.nn as nn

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import nested_width_net as nwn  # noqa: E402

from automl_package.enums import LayerSelectionMethod  # noqa: E402
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN  # noqa: E402

LOGVAR_HEAD_PATH_SUBSTRING = "logvar"  # nested_width_net.py's NestedWidthNet/IndependentWidthNet attribute name


# ---------------------------------------------------------------------------
# Shape descriptors for families with no real class yet (see module docstring).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthNetShapeDescriptor:
    """Shape of the depth-net family's shared-blocks-plus-shared-head arm.

    Planned `NestedDepthNet` (`docs/plans/capacity_programme/depth.md` Task D2 — not built at S2
    authoring time). One `input_size -> hidden_size` block, `max_depth - 1` chained
    `hidden_size -> hidden_size` blocks, and ONE shared `hidden_size -> output_size` readout
    applied after the routed depth's blocks (D2: "shared blocks + ONE shared output head over
    every depth prefix"). D2's other two planned arms (per-depth heads, independent nets) are not
    represented here — this descriptor covers only the shared-head arm; extend when the real
    classes land instead of guessing their per-arm deltas.
    """

    input_size: int
    hidden_size: int
    output_size: int
    max_depth: int


@dataclass(frozen=True)
class MoEShapeDescriptor:
    """Shape of the planned MoE family.

    `MoERegressionNet` (`docs/plans/capacity_programme/flexnn-moe.md` Task M2 — not built at S2
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
    dominated by the matmul term) are NOT separately counted — a stated convention of this module,
    not a claim about any external profiler's convention.
    """
    return int(in_features) * int(out_features)


def _linear_params(in_features: int, out_features: int) -> int:
    """Trainable parameter count of one `Linear(in_features, out_features)`: weight + bias."""
    return int(in_features) * int(out_features) + int(out_features)


def _sequential_linear_macs(seq: nn.Sequential) -> int:
    """Sums `_linear_macs` over every `nn.Linear` directly inside a `Sequential`.

    Non-`Linear` layers (activation, `BatchNorm1d`, `Dropout`) contribute no matmul and are
    skipped — this makes the helper correct for both FlexNN's `hidden_layers_blocks[i]` (Linear +
    optional BatchNorm + activation + optional Dropout) and its `n_predictor` (alternating Linear
    + activation) without needing a separate formula for each.
    """
    return sum(_linear_macs(m.in_features, m.out_features) for m in seq if isinstance(m, nn.Linear))


# ---------------------------------------------------------------------------
# param_count — one generic nn.Module implementation covers every width-net class and the FlexNN
# module; the two shape descriptors get explicit analytic overloads.
# ---------------------------------------------------------------------------


@functools.singledispatch
def param_count(net: object, path_filter: str | None = None) -> int:
    """Trainable parameter count of `net`.

    Args:
        net: an `nn.Module` (any width-net class from `nested_width_net.py`, or the FlexNN
            module `FlexibleHiddenLayersNN.FlexibleNNModule`), or one of this module's shape
            descriptors (`DepthNetShapeDescriptor`, `MoEShapeDescriptor`) for a family whose real
            class does not exist yet.
        path_filter: if given, EXCLUDES any parameter whose dotted name contains this substring
            (e.g. `LOGVAR_HEAD_PATH_SUBSTRING`, to drop the width nets' unused-under-MSE logvar
            heads). No effect on the shape-descriptor overloads — they have no named parameters
            to filter; the argument is accepted there for signature symmetry only.

    Returns:
        Total trainable (`requires_grad=True`) parameter count, honoring `path_filter` if given.

    Raises:
        TypeError: `net`'s type has no registered accounting implementation.
    """
    del path_filter  # unused in the dispatch stub — always raises before it would be read
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
# executed_flops — one registration per family; see module docstring for why width nets can't
# share a single generic implementation the way param_count does.
# ---------------------------------------------------------------------------


@functools.singledispatch
def executed_flops(net: object, config: int) -> int:
    """Analytic multiply-add (MAC) count of one forward pass of `net` AT ITS ROUTED CAPACITY.

    Not a timed benchmark (S2 spec is analytic-only). "Routed capacity" (`config`) means: width
    nets -> the routed width `k`; FlexNN -> the selected depth `d`; the depth-net descriptor ->
    its routed depth `d`; the MoE descriptor -> `top_k` (experts actually executed).

    Args:
        net: one of the width-net classes, the FlexNN module, or one of this module's shape
            descriptors for a family with no real class yet.
        config: the routed capacity level (see above); family-specific range checks raise
            `ValueError` if out of bounds.

    Returns:
        Multiply-add count for that one forward call at the given routed capacity.

    Raises:
        TypeError: `net`'s type has no registered accounting implementation.
    """
    del config  # unused in the dispatch stub — always raises before it would be read
    raise TypeError(f"executed_flops: no accounting implementation registered for {type(net).__name__}")


@executed_flops.register(nwn.NestedWidthNet)
def _executed_flops_nested_width(net: nwn.NestedWidthNet, config: int) -> int:
    """Routed width-`config` MACs for `NestedWidthNet` (shared trunk + ONE shared mean head).

    The prefix property (`nested_width_net.py` selftests (a)/(b)) guarantees hidden nodes
    `>= config` never influence a width-`config` output — they are zeroed before both readouts in
    `forward_width`, so an efficient width-`config` DEPLOYMENT needs only the first `config` trunk
    output rows and the first `config` input columns of `mean_head`, NOT the full `w_max`-wide
    matmuls `forward_width`/`all_widths_forward` literally compute (they compute the full width
    for masking-consistency, not for efficiency). `logvar_head` is excluded unconditionally — see
    module docstring (MASTER Decision 2).
    """
    k = config
    if not (1 <= k <= net.w_max):
        raise ValueError(f"config={k} out of range [1, {net.w_max}]")
    return _linear_macs(net.trunk.in_features, k) + _linear_macs(k, net.mean_head.out_features)


@executed_flops.register(nwn.SharedTrunkPerWidthHeadNet)
def _executed_flops_shared_trunk_per_width_head(net: nwn.SharedTrunkPerWidthHeadNet, config: int) -> int:
    """Routed width-`config` MACs for `SharedTrunkPerWidthHeadNet` (shared trunk, per-width heads).

    Same trunk-slicing argument as `NestedWidthNet` (the prefix property is trunk-level, not
    readout-level, so it holds for any head reading the masked hidden vector), then width-
    `config`'s OWN `mean_heads[config - 1]`, sliced to its own first `config` input columns by the
    same masked-to-zero argument. No logvar branch — this class's `log_var` is a dummy zero
    tensor, never computed (class docstring, `nested_width_net.py:236`).
    """
    k = config
    if not (1 <= k <= net.w_max):
        raise ValueError(f"config={k} out of range [1, {net.w_max}]")
    head = net.mean_heads[k - 1]
    return _linear_macs(net.trunk.in_features, k) + _linear_macs(k, head.out_features)


@executed_flops.register(nwn.IndependentWidthNet)
def _executed_flops_independent_width(net: nwn.IndependentWidthNet, config: int) -> int:
    """Routed width-`config` MACs for `IndependentWidthNet` (K disjoint sub-nets, no sharing).

    No slicing trick needed here (unlike the shared-trunk classes above) — sub-net `config` is
    already sized exactly `config`, with literal full compute: `trunk = Linear(1, config)` then
    `mean_head = Linear(config, 1)`. `logvar_head` excluded unconditionally — module docstring.
    """
    k = config
    if not (1 <= k <= net.w_max):
        raise ValueError(f"config={k} out of range [1, {net.w_max}]")
    sub = net.subnets[k - 1]
    trunk_lin, mean_lin = sub["trunk"], sub["mean_head"]
    return _linear_macs(trunk_lin.in_features, trunk_lin.out_features) + _linear_macs(mean_lin.in_features, mean_lin.out_features)


@executed_flops.register(nwn.SharedReadoutPerWidthAffineNet)
def _executed_flops_shared_readout_per_width_affine(net: nwn.SharedReadoutPerWidthAffineNet, config: int) -> int:
    """Routed width-`config` MACs for `SharedReadoutPerWidthAffineNet` (shared readout + per-width affine).

    Same trunk- and shared-`mean_head`-slicing argument as `NestedWidthNet`, plus the width-
    `config` affine `a_k * mean + c_k`: one multiply per sample (the shift-add is excluded under
    this module's bias convention, same as every Linear layer here) — `+1` MAC.
    """
    k = config
    if not (1 <= k <= net.w_max):
        raise ValueError(f"config={k} out of range [1, {net.w_max}]")
    affine_scale_mac = 1
    return _linear_macs(net.trunk.in_features, k) + _linear_macs(k, net.mean_head.out_features) + affine_scale_mac


@executed_flops.register(FlexibleHiddenLayersNN.FlexibleNNModule)
def _executed_flops_flexnn(net: FlexibleHiddenLayersNN.FlexibleNNModule, config: int) -> int:
    """Routed depth-`config` MACs for FlexNN's inner module.

    Mirrors `hard_forward`'s ACTUAL compute-saving code path
    (`models/flexible_neural_network.py:128-159`): `blocks[0..config-1]` run in sequence (depth
    blocks chain through activations — no width-style slicing trick, they are simply not entered
    at all past the routed depth), then `output_layer` runs once. If `n_predictor` exists (dynamic
    depth selection is on), `hard_forward` calls it on EVERY input to decide where to route — so
    its MACs are added unconditionally, not only at `config` depth.
    """
    d = config
    max_hidden_layers = len(net.hidden_layers_blocks)
    if not (1 <= d <= max_hidden_layers):
        raise ValueError(f"config={d} out of range [1, {max_hidden_layers}]")
    total = sum(_sequential_linear_macs(net.hidden_layers_blocks[i]) for i in range(d))
    total += _linear_macs(net.output_layer.in_features, net.output_layer.out_features)
    if net.n_predictor is not None:
        total += _sequential_linear_macs(net.n_predictor)
    return total


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
    — no matmul) run.
    """
    top_k = config
    if not (1 <= top_k <= net.n_experts):
        raise ValueError(f"config={top_k} out of range [1, {net.n_experts}]")
    router_macs = _linear_macs(net.d_in, net.n_experts)
    per_expert_macs = _linear_macs(net.d_in, net.expert_hidden) + _linear_macs(net.expert_hidden, 1)
    return router_macs + top_k * per_expert_macs


# ---------------------------------------------------------------------------
# Selftest -- hand-computed known-answer checks, one small config per family. MUST pass before
# any real read (`nested_width_net.py`'s selftest convention).
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Hand-computed known-answer checks, one small config per family.

    Every assertion's expected value is derived in its inline comment from the class's actual
    `Linear` shapes, not copied from this module's own formulas — so a bug shared between the
    formula and the check would still be caught.
    """
    results: list[tuple[str, bool]] = []

    # --- Width family: nested_width_net.py, w_max=4, routed width k=2 ---
    net_nested = nwn.NestedWidthNet(w_max=4)
    # trunk Linear(1,4): 1*4+4=8 params. mean_head Linear(4,1): 4*1+1=5. logvar_head: 5. all=18.
    expected_all_params = 8 + 5 + 5
    results.append(("NestedWidthNet param_count (all)", param_count(net_nested) == expected_all_params))
    # exclude logvar_head's 5 params.
    expected_mean_only_params = expected_all_params - 5
    results.append(("NestedWidthNet param_count (logvar excluded)", param_count(net_nested, path_filter=LOGVAR_HEAD_PATH_SUBSTRING) == expected_mean_only_params))
    # k=2: trunk sliced to Linear(1,2) -> 1*2=2 MACs; mean_head sliced to Linear(2,1) -> 2*1=2 MACs.
    expected_nested_flops_k2 = 2 + 2
    results.append(("NestedWidthNet executed_flops(k=2)", executed_flops(net_nested, 2) == expected_nested_flops_k2))
    try:
        executed_flops(net_nested, 0)
        oob_rejected = False
    except ValueError:
        oob_rejected = True
    results.append(("NestedWidthNet executed_flops rejects out-of-range config", oob_rejected))

    net_head = nwn.SharedTrunkPerWidthHeadNet(w_max=4)
    # trunk: 1*4+4=8. 4x mean_heads Linear(4,1): (4*1+1)=5 each -> 20. total (no logvar head).
    expected_head_params = 8 + 4 * 5
    results.append(("SharedTrunkPerWidthHeadNet param_count", param_count(net_head) == expected_head_params))
    # k=2: trunk sliced (1,2)=2 MACs + head sliced (2,1)=2 MACs.
    expected_head_flops_k2 = 2 + 2
    results.append(("SharedTrunkPerWidthHeadNet executed_flops(k=2)", executed_flops(net_head, 2) == expected_head_flops_k2))

    net_indep = nwn.IndependentWidthNet(w_max=4)
    # sub k=1: trunk(1,1)=1*1+1=2, mean(1,1)=1*1+1=2, logvar=2 -> 6.
    # sub k=2: trunk(1,2)=1*2+2=4, mean(2,1)=2*1+1=3, logvar=3 -> 10.
    # sub k=3: trunk(1,3)=1*3+3=6, mean(3,1)=3*1+1=4, logvar=4 -> 14.
    # sub k=4: trunk(1,4)=1*4+4=8, mean(4,1)=4*1+1=5, logvar=5 -> 18.
    expected_indep_all_params = 6 + 10 + 14 + 18
    expected_indep_logvar_only_params = 2 + 3 + 4 + 5
    expected_indep_mean_only_params = expected_indep_all_params - expected_indep_logvar_only_params
    results.append(("IndependentWidthNet param_count (all)", param_count(net_indep) == expected_indep_all_params))
    results.append(("IndependentWidthNet param_count (logvar excluded)", param_count(net_indep, path_filter=LOGVAR_HEAD_PATH_SUBSTRING) == expected_indep_mean_only_params))
    # k=2 sub-net: trunk Linear(1,2)=1*2=2 MACs, mean_head Linear(2,1)=2*1=2 MACs.
    expected_indep_flops_k2 = 2 + 2
    results.append(("IndependentWidthNet executed_flops(k=2)", executed_flops(net_indep, 2) == expected_indep_flops_k2))

    net_affine = nwn.SharedReadoutPerWidthAffineNet(w_max=4)
    # trunk: 8. mean_head Linear(4,1): 5. affine_scale (4,): 4. affine_bias (4,): 4.
    expected_affine_params = 8 + 5 + 4 + 4
    results.append(("SharedReadoutPerWidthAffineNet param_count", param_count(net_affine) == expected_affine_params))
    # k=2: trunk sliced (1,2)=2 + head sliced (2,1)=2 + affine scale=1.
    expected_affine_flops_k2 = 2 + 2 + 1
    results.append(("SharedReadoutPerWidthAffineNet executed_flops(k=2)", executed_flops(net_affine, 2) == expected_affine_flops_k2))

    # --- FlexNN family: models/flexible_neural_network.py, hidden_size=4, max_hidden_layers=2 ---
    flex_no_pred = FlexibleHiddenLayersNN(
        input_size=1, hidden_size=4, output_size=1, max_hidden_layers=2, layer_selection_method=LayerSelectionMethod.NONE, n_predictor_layers=0
    )
    flex_no_pred.build_model()
    module_no_pred = flex_no_pred.model
    # block0 Linear(1,4): 1*4+4=8. block1 Linear(4,4): 4*4+4=20. output_layer Linear(4,1): 4*1+1=5.
    expected_flexnn_no_pred_params = 8 + 20 + 5
    results.append(("FlexNN (no predictor) param_count", param_count(module_no_pred) == expected_flexnn_no_pred_params))
    # d=1: block0 (1,4)=4 MACs + output (4,1)=4 MACs.
    expected_flexnn_no_pred_flops_d1 = 4 + 4
    results.append(("FlexNN (no predictor) executed_flops(d=1)", executed_flops(module_no_pred, 1) == expected_flexnn_no_pred_flops_d1))
    # d=2: block0=4 + block1 (4,4)=16 + output=4.
    expected_flexnn_no_pred_flops_d2 = 4 + 16 + 4
    results.append(("FlexNN (no predictor) executed_flops(d=2)", executed_flops(module_no_pred, 2) == expected_flexnn_no_pred_flops_d2))

    flex_pred = FlexibleHiddenLayersNN(
        input_size=1, hidden_size=4, output_size=1, max_hidden_layers=2, layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX, n_predictor_layers=1
    )
    flex_pred.build_model()
    module_pred = flex_pred.model
    # n_predictor: predictor_hidden_size=max(128,4)=128 (flexible_neural_network.py:96, hardcoded floor).
    # Linear(1,128): 1*128=128 MACs. Linear(128,2) [out=max_hidden_layers]: 128*2=256 MACs.
    expected_predictor_macs = 128 + 128 * 2
    # d=1: block0 (1,4)=4 + output (4,1)=4 = 8 main-path MACs, plus the predictor (always runs).
    expected_flexnn_pred_flops_d1 = 4 + 4 + expected_predictor_macs
    results.append(("FlexNN (with predictor) executed_flops(d=1)", executed_flops(module_pred, 1) == expected_flexnn_pred_flops_d1))
    # d=2: block0=4 + block1(4,4)=16 + output=4 = 24 main-path, plus the predictor.
    expected_flexnn_pred_flops_d2 = 4 + 16 + 4 + expected_predictor_macs
    results.append(("FlexNN (with predictor) executed_flops(d=2)", executed_flops(module_pred, 2) == expected_flexnn_pred_flops_d2))

    # --- Depth-net shape descriptor (docs/plans/capacity_programme/depth.md Task D2, not built yet) ---
    depth_desc = DepthNetShapeDescriptor(input_size=2, hidden_size=4, output_size=1, max_depth=3)
    # block0 (2,4): 2*4+4=12. 2x chained (4,4): (4*4+4)=20 each -> 40. output head (4,1): 4*1+1=5.
    expected_depth_desc_params = 12 + 2 * 20 + 5
    results.append(("DepthNetShapeDescriptor param_count", param_count(depth_desc) == expected_depth_desc_params))
    # d=2: block0 (2,4)=8 MACs + 1 chained block (4,4)=16 MACs + output (4,1)=4 MACs.
    expected_depth_desc_flops_d2 = 8 + 16 + 4
    results.append(("DepthNetShapeDescriptor executed_flops(d=2)", executed_flops(depth_desc, 2) == expected_depth_desc_flops_d2))

    # --- MoE shape descriptor (docs/plans/capacity_programme/flexnn-moe.md Task M2, not built yet) ---
    moe_desc = MoEShapeDescriptor(d_in=3, expert_hidden=4, n_experts=8)
    # router (3,8): 3*8+8=32. per-expert: (3,4)=3*4+4=16 + (4,1)=4*1+1=5 = 21.
    expected_moe_desc_params = 32 + 8 * 21
    results.append(("MoEShapeDescriptor param_count", param_count(moe_desc) == expected_moe_desc_params))
    # top_k=2: router (3,8)=24 MACs (always full) + 2 * [(3,4)=12 + (4,1)=4].
    expected_moe_desc_flops_top2 = 24 + 2 * 16
    results.append(("MoEShapeDescriptor executed_flops(top_k=2)", executed_flops(moe_desc, 2) == expected_moe_desc_flops_top2))

    all_ok = all(isinstance(ok, bool) and ok for _, ok in results)
    for name, ok in results:
        print(f"[capacity_accounting selftest] {name}: {'PASS' if ok else 'FAIL'}")
    print(f"[capacity_accounting selftest] {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main() -> None:
    """Parses args and runs the selftest, or prints help (no standalone real-run mode)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Hand-computed known-answer checks, one small config per family.")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
