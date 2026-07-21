"""Characterization test for the FP-1 move (capacity-programme
`docs/plans/capacity_programme/flexnn-package.md` Task FP-1): the params/FLOPs accounting logic
moved from `automl_package/examples/capacity_accounting.py` into
`automl_package/utils/capacity_accounting.py`, leaving a re-export shim at the old location.

Two things must hold, permanently:
1. The shim's `param_count`/`executed_flops` are the SAME objects as the package's -- not a fork,
   not a copy -- so every registration (package-side FlexNN/FlexibleWidthNN, examples-side the
   four `nested_width_net.py` classes) lands on one shared dispatcher regardless of which import
   path a caller used.
2. Every known-answer check the pre-move module carried still holds, byte-identical, from BOTH
   import paths. These numbers are copied from the pre-move `capacity_accounting.py`'s own
   `run_selftest()` (unchanged formulas, unchanged expected values) -- reproduced here as a
   permanent pytest regression guard instead of only a scripted `--selftest` CLI check. Expected
   values are computed into named locals (not literals) before the assertion, matching that
   selftest's own style.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "automl_package", "examples"))
import nested_width_net as nwn

from automl_package.enums import LayerSelectionMethod
from automl_package.examples import capacity_accounting as shim
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.flexible_width_network import FlexibleWidthNN
from automl_package.utils import capacity_accounting as pkg


class TestShimIsPackageNotFork:
    """The examples-side shim must re-export the SAME dispatchers, not copies."""

    def test_executed_flops_identity(self):
        assert shim.executed_flops is pkg.executed_flops

    def test_param_count_identity(self):
        assert shim.param_count is pkg.param_count

    def test_shape_descriptors_identity(self):
        assert shim.DepthNetShapeDescriptor is pkg.DepthNetShapeDescriptor
        assert shim.MoEShapeDescriptor is pkg.MoEShapeDescriptor

    def test_logvar_constant_identity(self):
        expected = "logvar"
        assert shim.LOGVAR_HEAD_PATH_SUBSTRING == pkg.LOGVAR_HEAD_PATH_SUBSTRING == expected


class TestKnownAnswersUnchanged:
    """Every hand-computed known answer from the pre-move selftest, unchanged.

    Imported through `pkg` (the new home); `TestShimIsPackageNotFork` above already establishes
    that `shim.executed_flops is pkg.executed_flops`, so checking one checks both import paths.
    """

    def test_nested_width_net(self):
        net = nwn.NestedWidthNet(w_max=4)
        # trunk Linear(1,4): 1*4+4=8. mean_head Linear(4,1): 4*1+1=5. logvar_head: 5. all=18.
        expected_all_params = 8 + 5 + 5
        expected_mean_only_params = expected_all_params - 5
        assert pkg.param_count(net) == expected_all_params
        assert pkg.param_count(net, path_filter=pkg.LOGVAR_HEAD_PATH_SUBSTRING) == expected_mean_only_params
        # k=2: trunk sliced (1,2)=2 MACs + mean_head sliced (2,1)=2 MACs.
        expected_flops_k2 = 2 + 2
        assert pkg.executed_flops(net, 2) == expected_flops_k2
        with pytest.raises(ValueError, match="out of range"):
            pkg.executed_flops(net, 0)

    def test_shared_trunk_per_width_head_net(self):
        net = nwn.SharedTrunkPerWidthHeadNet(w_max=4)
        # trunk: 1*4+4=8. 4x mean_heads Linear(4,1)=5 each -> 20. no logvar head.
        expected_params = 8 + 4 * 5
        assert pkg.param_count(net) == expected_params
        # k=2: trunk sliced (1,2)=2 + head sliced (2,1)=2.
        expected_flops_k2 = 2 + 2
        assert pkg.executed_flops(net, 2) == expected_flops_k2

    def test_independent_width_net(self):
        net = nwn.IndependentWidthNet(w_max=4)
        # sub k=1: trunk(1,1)=2, mean(1,1)=2, logvar=2 -> 6. k=2: 4+3+3=10. k=3: 6+4+4=14. k=4: 8+5+5=18.
        expected_all_params = 6 + 10 + 14 + 18
        expected_logvar_only_params = 2 + 3 + 4 + 5
        expected_mean_only_params = expected_all_params - expected_logvar_only_params
        assert pkg.param_count(net) == expected_all_params
        assert pkg.param_count(net, path_filter=pkg.LOGVAR_HEAD_PATH_SUBSTRING) == expected_mean_only_params
        # k=2 sub-net: trunk Linear(1,2)=2 MACs, mean_head Linear(2,1)=2 MACs.
        expected_flops_k2 = 2 + 2
        assert pkg.executed_flops(net, 2) == expected_flops_k2

    def test_shared_readout_per_width_affine_net(self):
        net = nwn.SharedReadoutPerWidthAffineNet(w_max=4)
        # trunk: 8. mean_head Linear(4,1): 5. affine_scale (4,): 4. affine_bias (4,): 4.
        expected_params = 8 + 5 + 4 + 4
        assert pkg.param_count(net) == expected_params
        # k=2: trunk sliced=2 + head sliced=2 + affine scale=1.
        expected_flops_k2 = 2 + 2 + 1
        assert pkg.executed_flops(net, 2) == expected_flops_k2

    def test_flexnn_no_predictor(self):
        model = FlexibleHiddenLayersNN(
            input_size=1, hidden_size=4, output_size=1, max_hidden_layers=2, layer_selection_method=LayerSelectionMethod.NONE, n_predictor_layers=0
        )
        model.build_model()
        net = model.model
        # block0 Linear(1,4): 8. block1 Linear(4,4): 20. output_layer Linear(4,1): 5.
        expected_params = 8 + 20 + 5
        assert pkg.param_count(net) == expected_params
        # d=1: block0(1,4)=4 + output(4,1)=4.
        expected_flops_d1 = 4 + 4
        assert pkg.executed_flops(net, 1) == expected_flops_d1
        # d=2: block0=4 + block1(4,4)=16 + output=4.
        expected_flops_d2 = 4 + 16 + 4
        assert pkg.executed_flops(net, 2) == expected_flops_d2

    def test_flexnn_with_predictor(self):
        model = FlexibleHiddenLayersNN(
            input_size=1, hidden_size=4, output_size=1, max_hidden_layers=2, layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX, n_predictor_layers=1
        )
        model.build_model()
        net = model.model
        # n_predictor: Linear(1,128)=128 MACs, Linear(128,2)=256 MACs, always runs.
        expected_predictor_macs = 128 + 128 * 2
        # d=1: block0(1,4)=4 + output(4,1)=4 = 8 main-path, plus predictor.
        expected_flops_d1 = 4 + 4 + expected_predictor_macs
        assert pkg.executed_flops(net, 1) == expected_flops_d1
        # d=2: block0=4 + block1(4,4)=16 + output=4 = 24 main-path, plus predictor.
        expected_flops_d2 = 4 + 16 + 4 + expected_predictor_macs
        assert pkg.executed_flops(net, 2) == expected_flops_d2

    def test_depth_net_shape_descriptor(self):
        desc = pkg.DepthNetShapeDescriptor(input_size=2, hidden_size=4, output_size=1, max_depth=3)
        # block0 (2,4)=12. 2x chained (4,4)=20 each -> 40. output head (4,1)=5.
        expected_params = 12 + 2 * 20 + 5
        assert pkg.param_count(desc) == expected_params
        # d=2: block0(2,4)=8 + 1 chained block(4,4)=16 + output(4,1)=4.
        expected_flops_d2 = 8 + 16 + 4
        assert pkg.executed_flops(desc, 2) == expected_flops_d2

    def test_moe_shape_descriptor(self):
        desc = pkg.MoEShapeDescriptor(d_in=3, expert_hidden=4, n_experts=8)
        # router (3,8)=32. per-expert: (3,4)=16 + (4,1)=5 = 21.
        expected_params = 32 + 8 * 21
        assert pkg.param_count(desc) == expected_params
        # top_k=2: router (3,8)=24 MACs (always full) + 2 * [(3,4)=12 + (4,1)=4].
        expected_flops_top2 = 24 + 2 * 16
        assert pkg.executed_flops(desc, 2) == expected_flops_top2


class TestFlexibleWidthNNRegistration:
    """`FlexibleWidthNN.FlexibleWidthNNModule` registers against the SAME `executed_flops` used
    by every other family -- exercised via `fit_router`'s own default `cost_fn` path (mirrors
    `tests/test_flexible_width_network.py::test_fit_router_default_cost_fn_uses_s2_executed_flops`,
    reused here rather than re-deriving `FlexibleWidthNN`'s construction args (ladder rung 2)).
    """

    def test_executed_flops_dispatches_on_flexible_width_module(self):
        import numpy as np

        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 1)).astype(np.float32)
        y = (2.0 * x[:, 0] + rng.normal(scale=0.05, size=40)).astype(np.float32)

        model = FlexibleWidthNN(widths=[2, 4], max_epochs=5)
        model.fit(x, y)
        router = model.fit_router(x, y)

        for i, width in enumerate(model.widths):
            assert router.costs_[i] == pytest.approx(pkg.executed_flops(model.model, width))
