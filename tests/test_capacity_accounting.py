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

import math
import os
import sys

import numpy as np
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
        # GUMBEL_SOFTMAX is RETIRED under the nested ladder (MASTER Decision 29); this test's
        # subject is executed_flops' predictor-MAC accounting, which only exists when
        # n_predictor_layers > 0 -- both surviving members (NONE, NESTED) require
        # n_predictor_layers == 0, so a retired member plus the escape hatch is the only way left
        # to construct a predictor-bearing net at all. Not a claim that GUMBEL_SOFTMAX itself works.
        model = FlexibleHiddenLayersNN(
            input_size=1,
            hidden_size=4,
            output_size=1,
            max_hidden_layers=2,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            n_predictor_layers=1,
            allow_retired_capacity_selection=True,
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


class TestEndToEndSelectionCost:
    """WSEL-5 (`docs/plans/capacity_programme/width.md`): `global_cheap_cost`/`per_input_cost`/
    `global_sweep_cost` wire width.md Section 1's three models (W-SHARED/W-PERINPUT/W-SWEEP) onto
    the FP-9 selection-cost primitives (`router_fit_cost`/`held_out_read_cost`/`sweep_cost`), each
    producing one FINITE total (training + selection) rather than the primitives in isolation.
    Hand-computed expected values follow this file's own `TestKnownAnswersUnchanged` convention:
    named locals, not bare literals.
    """

    def test_global_cheap_cost_hand_computed(self):
        desc = pkg.MoEShapeDescriptor(d_in=2, expert_hidden=3, n_experts=4)
        # top_k=1: router(2,4)=8 + expert[(2,3)=6 + (3,1)=3] = 17. top_k=2: 8 + 2*9 = 26.
        expected_flops_top1 = 2 * 4 + (2 * 3 + 3 * 1)
        expected_flops_top2 = 2 * 4 + 2 * (2 * 3 + 3 * 1)
        n_samples = 5
        expected_selection_macs = n_samples * (expected_flops_top1 + expected_flops_top2)
        training_macs = 999  # caller-supplied, already-incurred training cost (see module note)

        result = pkg.global_cheap_cost(training_macs, desc, [1, 2], n_samples=n_samples)

        assert result.training_macs == training_macs
        assert result.selection_macs == expected_selection_macs
        assert result.total_macs == training_macs + expected_selection_macs

    def test_per_input_cost_hand_computed(self):
        # in_dim=2 -> hidden=(3,) -> n_capacities=4 MLP: forward = 2*3 + 3*4 = 18 MACs/sample.
        expected_forward_macs = 2 * 3 + 3 * 4
        expected_backward_forward_ratio = 2  # grad-input + grad-weight matmuls, each ~1x forward
        n_samples, n_epochs = 10, 5
        expected_selection_macs = n_epochs * n_samples * expected_forward_macs * (1 + expected_backward_forward_ratio)
        training_macs = 123

        result = pkg.per_input_cost(training_macs, in_dim=2, n_capacities=4, n_samples=n_samples, n_epochs=n_epochs, hidden=(3,))

        assert result.training_macs == training_macs
        assert result.selection_macs == expected_selection_macs
        assert result.total_macs == training_macs + expected_selection_macs

    def test_global_sweep_cost_hand_computed(self):
        desc = pkg.MoEShapeDescriptor(d_in=2, expert_hidden=3, n_experts=4)
        expected_flops_top1 = 2 * 4 + (2 * 3 + 3 * 1)
        expected_flops_top2 = 2 * 4 + 2 * (2 * 3 + 3 * 1)
        n_train_samples, n_epochs, backward_forward_ratio = 3, 2, 2
        expected_training_macs = n_epochs * n_train_samples * (1 + backward_forward_ratio) * (expected_flops_top1 + expected_flops_top2)
        n_selection_samples = 7
        expected_selection_macs = n_selection_samples * (expected_flops_top1 + expected_flops_top2)

        result = pkg.global_sweep_cost(desc, [1, 2], n_train_samples=n_train_samples, n_epochs=n_epochs, n_selection_samples=n_selection_samples)

        assert result.training_macs == expected_training_macs
        assert result.selection_macs == expected_selection_macs
        assert result.total_macs == expected_training_macs + expected_selection_macs

    def test_all_three_width_models_return_finite_total_cost_including_selection(self):
        """width.md Section 1's three width models, each costed end-to-end off ONE real `FlexibleWidthNN`."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 1)).astype(np.float32)
        y = (2.0 * x[:, 0] + rng.normal(scale=0.05, size=40)).astype(np.float32)

        model = FlexibleWidthNN(widths=[2, 4], max_epochs=5)
        model.fit(x, y)
        n_train_samples, n_epochs = 40, 5
        # W-SHARED/W-PERINPUT read off this SAME trained net (width.md Sec.1); `sweep_cost` over
        # its own widths grid is one legitimate caller-side way to price that training run -- see
        # `global_cheap_cost`'s module note for why the module itself does not assume this for you.
        training_macs = pkg.sweep_cost(model.model, model.widths, n_train_samples, n_epochs)

        shared = pkg.global_cheap_cost(training_macs, model.model, model.widths, n_samples=10)
        perinput = pkg.per_input_cost(training_macs, in_dim=1, n_capacities=len(model.widths), n_samples=10, n_epochs=300, hidden=(32, 32))
        sweep = pkg.global_sweep_cost(model.model, model.widths, n_train_samples=n_train_samples, n_epochs=n_epochs, n_selection_samples=10)

        for breakdown in (shared, perinput, sweep):
            assert breakdown.training_macs > 0
            assert breakdown.selection_macs > 0
            assert math.isfinite(breakdown.total_macs)
            assert breakdown.total_macs == breakdown.training_macs + breakdown.selection_macs
