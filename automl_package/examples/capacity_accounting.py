"""S2 -- the ONE source of params/FLOPs numbers for reports (b)/(c) and the strand-4 deploy bars.

Thin re-export shim over `automl_package.utils.capacity_accounting` -- the accounting logic itself
moved there in `docs/plans/capacity_programme/flexnn-package.md` Task FP-1 (the
`automl_package/examples/convergence.py` precedent: move the logic, leave a shim, keep every
existing script's `import capacity_accounting as ca` / `from automl_package.examples
.capacity_accounting import ...` resolving unchanged).

**What stays here, and why.** `executed_flops` dispatches on the four `nested_width_net.py` width
classes (`NestedWidthNet`, `SharedTrunkPerWidthHeadNet`, `IndependentWidthNet`,
`SharedReadoutPerWidthAffineNet`), imported by bare name below. Those classes don't move into the
package until FP-2, and the package module must import nothing from `examples/` -- so their
registrations stay here, registered onto the SAME `executed_flops` singledispatch function via its
public `.register()` (the ordinary `functools.singledispatch` extension mechanism; no bespoke
registration hook needed). FP-2's last step moves these four registrations into the package
alongside the classes, and this file drops to a pure re-export.

The scripted selftest CLI also stays here, since that is example-script behavior, not package
logic (same rationale as `convergence.py`'s selftest).

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_accounting.py --selftest
"""

from __future__ import annotations

import argparse
import os
import sys

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import nested_width_net as nwn  # noqa: E402

from automl_package.enums import LayerSelectionMethod  # noqa: E402
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN  # noqa: E402
from automl_package.utils.capacity_accounting import (  # noqa: E402
    LOGVAR_HEAD_PATH_SUBSTRING,
    DepthNetShapeDescriptor,
    MoEShapeDescriptor,
    _linear_macs,
    executed_flops,
    held_out_read_cost,
    param_count,
    router_fit_cost,
    sweep_cost,
)

__all__ = [
    "LOGVAR_HEAD_PATH_SUBSTRING",
    "DepthNetShapeDescriptor",
    "MoEShapeDescriptor",
    "executed_flops",
    "held_out_read_cost",
    "param_count",
    "router_fit_cost",
    "sweep_cost",
]


# ---------------------------------------------------------------------------
# executed_flops registrations for the four nested_width_net.py classes -- stay here until FP-2
# (see module docstring). Formulas unchanged from before the FP-1 move.
# ---------------------------------------------------------------------------


@executed_flops.register(nwn.NestedWidthNet)
def _executed_flops_nested_width(net: nwn.NestedWidthNet, config: int) -> int:
    """Routed width-`config` MACs for `NestedWidthNet` (shared trunk + ONE shared mean head).

    The prefix property (`nested_width_net.py` selftests (a)/(b)) guarantees hidden nodes
    `>= config` never influence a width-`config` output -- they are zeroed before both readouts in
    `forward_width`, so an efficient width-`config` DEPLOYMENT needs only the first `config` trunk
    output rows and the first `config` input columns of `mean_head`, NOT the full `w_max`-wide
    matmuls `forward_width`/`all_widths_forward` literally compute (they compute the full width
    for masking-consistency, not for efficiency). `logvar_head` is excluded unconditionally -- see
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
    same masked-to-zero argument. No logvar branch -- this class's `log_var` is a dummy zero
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

    No slicing trick needed here (unlike the shared-trunk classes above) -- sub-net `config` is
    already sized exactly `config`, with literal full compute: `trunk = Linear(1, config)` then
    `mean_head = Linear(config, 1)`. `logvar_head` excluded unconditionally -- module docstring.
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
    this module's bias convention, same as every Linear layer here) -- `+1` MAC.
    """
    k = config
    if not (1 <= k <= net.w_max):
        raise ValueError(f"config={k} out of range [1, {net.w_max}]")
    affine_scale_mac = 1
    return _linear_macs(net.trunk.in_features, k) + _linear_macs(k, net.mean_head.out_features) + affine_scale_mac


# ---------------------------------------------------------------------------
# Selftest -- hand-computed known-answer checks, one small config per family. MUST pass before
# any real read (`nested_width_net.py`'s selftest convention).
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Hand-computed known-answer checks, one small config per family.

    Every assertion's expected value is derived in its inline comment from the class's actual
    `Linear` shapes, not copied from this module's own formulas -- so a bug shared between the
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

    # --- Selection-cost accounting (capacity-programme Task FP-9.c) ---
    # The cost of CHOOSING a capacity, not of running one. Added here because `run_selftest` lives
    # in this shim while the functions live in the package module; applied by the root, since the
    # FP-9 worker's write set excludes `examples/` (single-writer discipline).
    # router MLP 3 -> (4,) -> 2: forward MACs = 3*4 + 4*2 = 20. Full batch: x10 samples x5 epochs,
    # x3 for forward+backward (backward ~ 2x forward).
    expected_router_fit = (3 * 4 + 4 * 2) * 10 * 5 * 3
    results.append(("router_fit_cost (selection)", router_fit_cost(in_dim=3, n_capacities=2, n_samples=10, n_epochs=5, hidden=(4,)) == expected_router_fit))
    # cheap held-out read over the MoE descriptor above at top_k in {1,2}: one forward per sample
    # per capacity, no backward. top_k=1: 24 + 16 = 40. top_k=2: 24 + 2*16 = 56.
    expected_held_out_read = 7 * ((24 + 16) + (24 + 2 * 16))
    results.append(("held_out_read_cost (selection)", held_out_read_cost(moe_desc, (1, 2), n_samples=7) == expected_held_out_read))
    # sweep: one dedicated model trained per capacity -- same per-capacity FLOPs, x samples x epochs x3.
    expected_sweep = 5 * 7 * (24 + 16) * 3 + 5 * 7 * (24 + 2 * 16) * 3
    results.append(("sweep_cost (selection)", sweep_cost(moe_desc, (1, 2), n_train_samples=7, n_epochs=5) == expected_sweep))

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
