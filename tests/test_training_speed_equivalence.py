"""The training-loop speed changes keep results bit-identical.

- ``shuffled_batches`` yields the same batches as ``DataLoader(TensorDataset(...), shuffle=True)`` and leaves the global
  random generator in the same state (so initialisation and dropout drawn later are unchanged).
- The vectorised L1/L2 sums give the same gradients as the per-parameter sums they replace.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from automl_package.utils.pytorch_utils import calculate_regularization_loss, shuffled_batches


@pytest.mark.parametrize("drop_last", [False, True])
def test_shuffled_batches_match_dataloader(drop_last):
    x = torch.arange(30, dtype=torch.float32).unsqueeze(1)
    y = -x
    torch.manual_seed(5)
    loader = DataLoader(TensorDataset(x, y), batch_size=7, shuffle=True, drop_last=drop_last)
    expected = [[b[0].clone(), b[1].clone()] for _ in range(2) for b in loader]
    state_after_loader = torch.get_rng_state()

    torch.manual_seed(5)
    got = [[x[rows], y[rows]] for _ in range(2) for rows in shuffled_batches(len(x), 7, drop_last=drop_last)]

    assert len(got) == len(expected)
    assert all(torch.equal(a, b) for pair_got, pair_expected in zip(got, expected, strict=True) for a, b in zip(pair_got, pair_expected, strict=True))
    assert torch.equal(torch.get_rng_state(), state_after_loader)


def test_vectorised_penalty_gradients_match_per_parameter_sums():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
    model[2].weight.requires_grad_(False)
    base = model(torch.randn(4, 3)).pow(2).mean()

    def per_parameter_penalty() -> torch.Tensor:  # the code the vectorised sums replaced (biases excluded, frozen weights skipped)
        l1, l2 = 0.0, 0.0
        for name, p in model.named_parameters():
            if p.requires_grad and "bias" not in name:
                l1 = l1 + p.view(-1).abs().sum()
                l2 = l2 + (p.view(-1) ** 2).sum()
        return base + 0.3 * l1 + 0.7 * l2

    kwargs = {"learn_regularization": False, "learned_regularization_type": None, "l1_lambda": 0.3, "l2_lambda": 0.7, "l1_log_lambda": None, "l2_log_lambda": None}
    trainable = [p for p in model.parameters() if p.requires_grad]
    expected, got = per_parameter_penalty(), calculate_regularization_loss(base_loss=base, model=model, **kwargs)
    torch.testing.assert_close(got, expected)  # same value up to summation order
    grads_expected = torch.autograd.grad(expected, trainable, retain_graph=True)
    grads_got = torch.autograd.grad(got, trainable)
    assert all(torch.equal(a, b) for a, b in zip(grads_got, grads_expected, strict=True))  # bit-identical gradients
