"""WSEL-16 Step 1 -- the stop-gradient identity test, BEFORE any training (`docs/plans/capacity_programme/
width.md` ~1592-1597).

`A_STOPGRAD`'s per-width loss is `sum_k MSE(detach(S_k - c_k) + c_k, y)`, where `S_k = b + sum_{j<=k}
c_j` is `NestedWidthNet`'s plain width-k running sum and `c_j = w_j * h_j` unit j's own contribution.
Algebraically `S_k - c_k = S_{k-1}`, so this is `detach(S_{k-1}) + c_k` per width --
`width_wsel16.stopgrad_all_widths_pred` computes the whole `(N, w_max)` table in ONE cumsum pass, no
python loop over widths.

Two properties must hold, on a fixed seed and a fixed `(64, 1)` input/target:

  (a) VALUE identity -- `detach()` changes gradients, not values, so the stop-gradient loss's VALUE must
      equal the plain summed loss's value exactly (`NestedWidthNet.all_widths_forward`, undetached).
  (b) GRADIENT identity -- unit 1's weight `w_1` receives gradient ONLY through the k=1 term (every other
      term's `S_{k-1}` -- which contains `c_1` for k > 1 -- is detached), so `d(stopgrad loss)/d(w_1)`
      must equal `d(MSE(S_1, y))/d(w_1)` computed from the width-1 term ALONE.

**Prove-it-fails** (`test_prove_it_fails_without_detach`): drops the `detach()` (i.e. reads the PLAIN
summed loss's gradient, `A_JOINT`'s own loss) and shows (b) FAILS there -- `w_1` is pulled by all 4
width terms in the tug-of-war the stop-gradient loss exists to break, not just the width-1 term. If this
test does NOT fail, (b)'s pass above would be meaningless (nothing would distinguish "detach worked"
from "the two quantities were always equal").

Gradients are compared in float64 (`tests/test_nested_width_single_trunk.py`'s precedent: float32
matmul reduction-order non-associativity produces ~1e-7 grad noise even for an algorithmically-exact
comparison -- this test wants EXACT equality, so it runs in double precision throughout).

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_stopgrad_width_loss.py -q
"""

from __future__ import annotations

import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "automl_package", "examples"))
import nested_width_net as nwn
import width_wsel16 as w16

_W_MAX = 4
_N = 64
_SEED = 0
_EXACT_TOL = 1e-12  # float64 exactness bar (test_nested_width_single_trunk.py's own convention)


def _fixed_net_and_data() -> tuple[nwn.NestedWidthNet, torch.Tensor, torch.Tensor]:
    """A fixed-seed `NestedWidthNet(w_max=4)` in float64, plus a fixed `(64, 1)` input/target."""
    torch.manual_seed(_SEED)
    net = nwn.NestedWidthNet(w_max=_W_MAX).double()
    gen = torch.Generator().manual_seed(_SEED)
    x = torch.randn(_N, 1, generator=gen, dtype=torch.float64)
    y = torch.randn(_N, generator=gen, dtype=torch.float64)
    return net, x, y


def _summed_loss(preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """`sum_k MSE(preds[:, k-1], y)` over every width -- the shared reduction both the stop-gradient and plain-sum losses use."""
    return ((preds - y.unsqueeze(1)) ** 2).mean(dim=0).sum()


def test_stopgrad_value_equals_plain_summed_loss() -> None:
    """(a) `detach()` changes gradients, not values: the stop-gradient loss VALUE equals the plain summed loss VALUE exactly (up to float64 reduction-order noise).

    `stopgrad_all_widths_pred` reassociates the SAME cumulative sum as `all_widths_forward`
    (`(A + bias) + contrib` vs `(A + contrib) + bias`) to get the per-width detach split, so the two are
    not computed by bit-identical instruction sequences -- IEEE-754 addition is commutative but NOT
    associative, so `torch.equal` is the wrong bar here (measured: max abs diff ~1.1e-16 at float64, pure
    reduction-order noise, not an algorithmic difference -- `test_nested_width_single_trunk.py`'s own
    precedent for this exact class of float non-associativity noise). `torch.allclose` at `_EXACT_TOL`
    is ~1e4x looser than the measured noise floor and still tight enough to catch a real algorithmic bug.
    """
    net, x, y = _fixed_net_and_data()
    stopgrad_preds = w16.stopgrad_all_widths_pred(net, x)
    plain_preds, _log_var = net.all_widths_forward(x)

    assert torch.allclose(stopgrad_preds, plain_preds, atol=_EXACT_TOL, rtol=0.0), "detach() must not change the per-width prediction VALUES"

    stopgrad_loss = _summed_loss(stopgrad_preds, y)
    plain_loss = _summed_loss(plain_preds, y)
    assert torch.allclose(stopgrad_loss, plain_loss, atol=_EXACT_TOL, rtol=0.0), (
        f"stopgrad loss value {stopgrad_loss.item()} != plain summed loss value {plain_loss.item()}"
    )


def test_stopgrad_w1_gradient_equals_width1_term_alone() -> None:
    """(b) `d(stopgrad total loss)/d(w_1)` equals `d(MSE(S_1, y))/d(w_1)` -- w_1 feels only the width-1 term."""
    net, x, y = _fixed_net_and_data()

    net.zero_grad()
    stopgrad_preds = w16.stopgrad_all_widths_pred(net, x)
    stopgrad_loss = _summed_loss(stopgrad_preds, y)
    stopgrad_loss.backward()
    grad_stopgrad_w1 = net.mean_head.weight.grad[0, 0].clone()

    net_alone = copy.deepcopy(net)
    net_alone.zero_grad()
    mean_1, _log_var_1 = net_alone.forward_width(x, 1)
    loss_width1_alone = ((mean_1.squeeze(1) - y) ** 2).mean()
    loss_width1_alone.backward()
    grad_width1_alone = net_alone.mean_head.weight.grad[0, 0].clone()

    print(f"[stopgrad identity] grad_stopgrad_w1={grad_stopgrad_w1.item():.12f}  grad_width1_alone={grad_width1_alone.item():.12f}")
    assert torch.allclose(grad_stopgrad_w1, grad_width1_alone, atol=_EXACT_TOL, rtol=0.0), (
        f"stopgrad d(loss)/d(w_1)={grad_stopgrad_w1.item()} != width-1-alone d(loss)/d(w_1)={grad_width1_alone.item()}"
    )


def test_prove_it_fails_without_detach() -> None:
    """Prove-it-fails: drop the detach (A_JOINT's plain summed loss) and show (b) FAILS -- w_1 is pulled by every width term, not just width 1."""
    net, x, y = _fixed_net_and_data()

    net.zero_grad()
    plain_preds, _log_var = net.all_widths_forward(x)  # NO detach -- A_JOINT's own loss.
    plain_loss = _summed_loss(plain_preds, y)
    plain_loss.backward()
    grad_plain_w1 = net.mean_head.weight.grad[0, 0].clone()

    net_alone = copy.deepcopy(net)
    net_alone.zero_grad()
    mean_1, _log_var_1 = net_alone.forward_width(x, 1)
    loss_width1_alone = ((mean_1.squeeze(1) - y) ** 2).mean()
    loss_width1_alone.backward()
    grad_width1_alone = net_alone.mean_head.weight.grad[0, 0].clone()

    print(f"[prove-it-fails] grad_plain_w1={grad_plain_w1.item():.12f}  grad_width1_alone={grad_width1_alone.item():.12f}")
    assert not torch.allclose(grad_plain_w1, grad_width1_alone, atol=_EXACT_TOL, rtol=0.0), (
        "expected the UNDETACHED plain-sum gradient to DIFFER from the width-1-alone gradient (the tug-of-war "
        "the stop-gradient loss exists to break) -- if they match, this test is not discriminating anything"
    )
