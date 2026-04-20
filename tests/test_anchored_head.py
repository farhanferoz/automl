"""Tests for AnchoredHead and its integration into SeparateHeadsRegressionModule (CT2)."""

import torch

from automl_package.enums import UncertaintyMethod
from automl_package.models.common.regression_heads import AnchoredHead, SeparateHeadsRegressionModule


class TestAnchoredHead:
    def _make_head(self, centroid: float = 2.5) -> AnchoredHead:
        torch.manual_seed(0)
        return AnchoredHead(centroid=centroid, hidden_size=16)

    def test_ct2a_anchor_at_p_equals_one(self):
        """h_i(p_i=1) == centroid to float precision."""
        centroid = 3.7
        head = self._make_head(centroid)
        p_one = torch.ones(4, 1)
        out = head(p_one)
        means = out[:, 0]
        # gate = 1 - 1 = 0 → mean = centroid exactly
        assert torch.allclose(means, torch.full((4,), centroid), atol=1e-6)

    def test_ct2b_formula_at_p_equals_zero(self):
        """h_i(p_i=0) == centroid + f(0)[0]."""
        centroid = 1.0
        head = self._make_head(centroid)
        p_zero = torch.zeros(1, 1)
        with torch.no_grad():
            f_out = head.f(p_zero)
            expected_mean = centroid + 1.0 * f_out[0, 0].item()
        out = head(p_zero)
        assert abs(out[0, 0].item() - expected_mean) < 1e-6

    def test_ct2c_gradients_flow_to_f_not_centroid(self):
        """Gradient flows to f parameters; centroid is a buffer (no gradient)."""
        head = self._make_head(2.0)
        p = torch.rand(8, 1)
        out = head(p)
        loss = out.sum()
        loss.backward()

        # f parameters should have gradients
        for name, param in head.named_parameters():
            assert param.grad is not None, f"param {name} has no gradient"

        # centroid is a buffer, not a parameter
        assert "centroid" not in dict(head.named_parameters())
        assert head.centroid.requires_grad is False

    def test_ct2d_module_construction_with_anchored_heads(self):
        """SeparateHeadsRegressionModule with use_anchored_heads=True creates k AnchoredHeads."""
        k = 5
        centroids = [float(i) for i in range(k)]
        module = SeparateHeadsRegressionModule(
            n_classes=k,
            regression_head_params={"hidden_size": 16},
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            regression_output_size=2,
            use_anchored_heads=True,
            centroids=centroids,
        )
        assert len(module.heads) == k
        for i, head in enumerate(module.heads):
            assert isinstance(head, AnchoredHead), f"head {i} is {type(head)}"
            assert abs(head.centroid.item() - centroids[i]) < 1e-6

    def test_forward_shape(self):
        """AnchoredHead produces [batch, 2] output."""
        head = self._make_head(0.0)
        p = torch.rand(12, 1)
        out = head(p)
        assert out.shape == (12, 2)

    def test_module_forward_with_anchored_heads(self):
        """SeparateHeadsRegressionModule forward works end-to-end with anchored heads."""
        k = 3
        centroids = [-1.0, 0.0, 1.0]
        module = SeparateHeadsRegressionModule(
            n_classes=k,
            regression_head_params={"hidden_size": 16},
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            regression_output_size=2,
            use_anchored_heads=True,
            centroids=centroids,
        )
        probs = torch.softmax(torch.randn(10, k), dim=-1)
        final_pred, per_head = module(probs, return_head_outputs=True)
        assert final_pred.shape == (10, 2)
        assert per_head.shape == (10, k, 2)
