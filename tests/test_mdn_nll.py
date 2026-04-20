"""Tests for the MDN NLL loss function (CT1)."""

import math

import torch

from automl_package.utils.losses import mdn_nll


def _gaussian_log_prob(y: float, mu: float, log_var: float) -> float:
    var = math.exp(log_var)
    return -0.5 * (math.log(2 * math.pi) + log_var + (y - mu) ** 2 / var)


class TestMdnNll:
    def test_ct1a_hand_computed_k2(self):
        """Hand-computed NLL for k=2 mixture matches mdn_nll output."""
        y = torch.tensor([1.0])
        probs = torch.tensor([[0.3, 0.7]])
        mus = torch.tensor([[0.0, 2.0]])
        log_vars = torch.tensor([[0.0, 0.0]])  # unit variance

        log_p0 = math.log(0.3) + _gaussian_log_prob(1.0, 0.0, 0.0)
        log_p1 = math.log(0.7) + _gaussian_log_prob(1.0, 2.0, 0.0)
        expected = -math.log(math.exp(log_p0) + math.exp(log_p1))

        result = mdn_nll(y, probs, mus, log_vars).item()
        assert abs(result - expected) < 1e-5

    def test_ct1b_degenerate_single_component(self):
        """When one p_j → 1, MDN NLL reduces to single Gaussian NLL."""
        y = torch.tensor([1.5])
        probs = torch.tensor([[1.0 - 1e-7, 1e-7]])
        mus = torch.tensor([[1.5, 0.0]])
        log_vars = torch.tensor([[0.5, 0.5]])

        mdn_val = mdn_nll(y, probs, mus, log_vars).item()
        single_nll = -_gaussian_log_prob(1.5, 1.5, 0.5)
        # Should be very close since the second component weight is negligible.
        assert abs(mdn_val - single_nll) < 0.05

    def test_ct1c_gradients_flow_to_all_inputs(self):
        """Gradients flow to probs, mus, and log_vars."""
        y = torch.randn(8)
        probs = torch.softmax(torch.randn(8, 3), dim=-1).requires_grad_(True)
        mus = torch.randn(8, 3, requires_grad=True)
        log_vars = torch.randn(8, 3, requires_grad=True)

        loss = mdn_nll(y, probs, mus, log_vars)
        loss.backward()

        assert probs.grad is not None
        assert probs.grad.abs().sum() > 0
        assert mus.grad is not None
        assert mus.grad.abs().sum() > 0
        assert log_vars.grad is not None
        assert log_vars.grad.abs().sum() > 0

    def test_ct1d_numerical_stability_extreme_log_vars(self):
        """No NaN or Inf for extreme log_var values."""
        y = torch.randn(10)
        probs = torch.softmax(torch.randn(10, 3), dim=-1)
        mus = torch.randn(10, 3)

        for extreme in [-20.0, 20.0]:
            log_vars = torch.full((10, 3), extreme)
            result = mdn_nll(y, probs, mus, log_vars)
            assert torch.isfinite(result), f"NaN/Inf with log_var={extreme}"

    def test_output_is_scalar(self):
        """Output is a scalar tensor."""
        y = torch.randn(16)
        probs = torch.softmax(torch.randn(16, 4), dim=-1)
        mus = torch.randn(16, 4)
        log_vars = torch.zeros(16, 4)
        result = mdn_nll(y, probs, mus, log_vars)
        assert result.shape == ()

    def test_y_shape_variants(self):
        """Accepts both [batch] and [batch, 1] y shapes."""
        probs = torch.softmax(torch.randn(8, 3), dim=-1)
        mus = torch.randn(8, 3)
        log_vars = torch.zeros(8, 3)
        y_1d = torch.randn(8)
        y_2d = y_1d.unsqueeze(1)
        assert abs(mdn_nll(y_1d, probs, mus, log_vars).item() - mdn_nll(y_2d, probs, mus, log_vars).item()) < 1e-6
