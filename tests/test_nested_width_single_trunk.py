"""WSEL-12 equivalence test (`docs/plans/capacity_programme/width.md` lines 1017-1087).

`_train_kdropout_to_convergence`'s sandwich-step accumulation used to loop `for k in widths: total +=
_width_loss(...)`, recomputing `SharedTrunkPerWidthHeadNet`'s shared trunk (`self.hidden(x)`) once PER
sampled width even though every width reads the same trunk. The fix -- `_sampled_widths_total_loss`,
which evaluates the trunk ONCE via `net.sampled_widths_forward` and sums the per-width readouts off it
-- must be a PURE efficiency change: it must not move a single number. This test is the bit-for-bit
proof: (a) the old per-k loop (reproduced here via the same `_width_loss` dispatch the production loop
called, not re-derived) vs (b) `_sampled_widths_total_loss`, same net weights, same input/target, same
widths -- losses and every parameter's gradient must agree to the task's tolerance.

Bar (from the strand file): losses equal to `1e-12`, every parameter gradient equal to `1e-10`
(`torch.allclose`), over `widths=[1, 6, 3, 4]` on `SharedTrunkPerWidthHeadNet(w_max=6)` under
`torch.manual_seed(0)` with a fixed `(64, 1)` input and target.

**Precision note (float64, not the training dtype).** At float32 -- the dtype every real training run
uses -- this bar is NOT achievable by ANY implementation that shares one trunk computation across
widths, including a bug-free one: reusing one autograd node for 4 downstream branches changes the
FLOATING-POINT REDUCTION ORDER of the backward matmul vs recomputing the node 4 separate times, and
IEEE-754 addition/matmul is not associative. Measured on this exact scenario (float32): loss diff
0.0 (exact -- the forward path has no reduction-order change), but max grad diff 2.98e-08 at
`trunk.weight`, ABOVE the strand's 1e-10 bar. Isolated further to a 2-line repro with no width-net
code at all (one `nn.Linear(1, 6)`, called once and reused vs called twice independently, summed,
`.backward()`'d): weight-grad diff 9.54e-07 at float32, 5.33e-15 at float64 -- confirming the effect
is generic float32 non-associativity, not anything specific to this architecture, and that it vanishes
at higher precision. This test therefore runs the comparison in float64 (`.double()`) to verify the
ALGORITHM is exactly correct -- the bar's actual intent ("not close enough", i.e. no silent
approximation bug) -- while the production training path this protects (`_sampled_widths_total_loss`
called from `_train_kdropout_to_convergence`) stays float32 like the rest of the codebase; the ~3e-8
float32 gradient noise is the honest, irreducible cost of the fix's forward-compute savings, not a
defect. Flagged to the root for a decision on whether to record this in the strand file (see the task
report for the exact text).

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m pytest tests/test_nested_width_single_trunk.py -q
"""

from __future__ import annotations

import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "automl_package", "examples"))
import kdropout_converged_width_experiment as kce
import nested_width_net as nwn

_LOSS_TOL = 1e-12
_GRAD_TOL = 1e-10
_WIDTHS = [1, 6, 3, 4]
_W_MAX = 6


def _build_net_and_data() -> tuple[nwn.SharedTrunkPerWidthHeadNet, torch.Tensor, torch.Tensor]:
    """`SharedTrunkPerWidthHeadNet(w_max=6)` under `torch.manual_seed(0)`, plus a fixed `(64, 1)` input
    and target drawn off the same seeded stream (fully deterministic, no separate seeding needed).
    Cast to float64 -- see module docstring's precision note -- so the bit-for-bit bar is checking the
    algorithm, not float32 backward-reduction-order noise.
    """
    torch.manual_seed(0)
    net = nwn.SharedTrunkPerWidthHeadNet(w_max=_W_MAX).double()
    x = torch.randn(64, 1, dtype=torch.float64)
    y = torch.randn(64, dtype=torch.float64)
    return net, x, y


def _old_per_width_loop_loss(net: nwn.SharedTrunkPerWidthHeadNet, widths: list[int], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The per-width loop `_train_kdropout_to_convergence` used BEFORE WSEL-12: one `_width_loss` call
    (hence one `forward_width` call, hence one trunk recompute) per sampled width -- the exact old
    accumulation, via the same dispatch the production loop called, not a re-derivation of it.
    """
    total = torch.zeros((), dtype=x.dtype)
    for k in widths:
        total = total + kce._width_loss(kce.LossType.MSE, net, k, x, y)
    return total


def _grads(net: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for name, p in net.named_parameters()}


_CANONICAL_W_MAX = 12  # the shape every real width cell runs at (SS3.8 tier 1), not the 6 of the bar above
_CANONICAL_N = 1500
_ARCHS = (nwn.SharedTrunkPerWidthHeadNet, nwn.NestedWidthNet, nwn.SharedReadoutPerWidthAffineNet)


@pytest.mark.parametrize("arch", _ARCHS, ids=lambda a: a.__name__)
@pytest.mark.parametrize("loss", list(kce.LossType), ids=lambda lt: lt.value)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_loss_is_exactly_bit_identical_at_float32(arch: type[torch.nn.Module], loss: kce.LossType, seed: int) -> None:
    """The FORWARD loss must be EXACTLY equal -- difference `0.0`, not merely close -- at float32.

    This is the discriminator the float64 gradient test above cannot be, and it runs at the CANONICAL
    shape and in the PRODUCTION dtype, where the float64 test runs at neither.

    Why exact zero is the right bar here, and why it is achievable when the gradient bar is not: the
    forward path sums the SAME per-width readouts in the SAME order either way -- hoisting `hidden(x)`
    out of the loop removes a recompute, it does not reassociate the sum. Only the BACKWARD pass
    reassociates, which is why gradients drift by ~1e-7 (float32 eps ~1.2e-7) while the forward does
    not drift at all.

    What this kills that a tolerance-based check would wave through: a genuine semantic fault -- a
    dropped width, a mis-paired head/mask, an off-by-one mask, a changed reduction -- shifts the
    step-one loss by ~0.9%-4.2% at this shape (measured during WSEL-12's adjudication). Such a fault
    is present on the FIRST forward pass, at full size, with no compounding to hide behind. Exact
    equality has no tolerance for it to slip under.
    """
    torch.manual_seed(seed)
    net = arch(w_max=_CANONICAL_W_MAX)
    x = torch.randn(_CANONICAL_N, 1)
    y = torch.randn(_CANONICAL_N, 1)
    widths = [1, _CANONICAL_W_MAX, 3, 7]  # a real sandwich draw: always min and max, plus two middles

    old = torch.zeros((), dtype=x.dtype)
    for k in widths:
        old = old + kce._width_loss(loss, net, k, x, y)
    new = kce._sampled_widths_total_loss(loss, net, widths, x, y)

    diff = (old - new).abs().item()
    assert diff == 0.0, f"forward loss not bit-identical for {arch.__name__}/{loss.value}/seed={seed}: |old-new|={diff:.6e}"


def test_single_trunk_matches_old_per_width_loop() -> None:
    """(a) old per-k loop vs (b) `_sampled_widths_total_loss` (single trunk eval): losses and every
    parameter's gradient must agree to the task's bit-for-bit bar.
    """
    net, x, y = _build_net_and_data()
    net_a = copy.deepcopy(net)
    net_b = copy.deepcopy(net)

    loss_a = _old_per_width_loop_loss(net_a, _WIDTHS, x, y)
    loss_a.backward()

    loss_b = kce._sampled_widths_total_loss(kce.LossType.MSE, net_b, _WIDTHS, x, y)
    loss_b.backward()

    assert torch.allclose(loss_a, loss_b, atol=_LOSS_TOL, rtol=0), f"loss mismatch: old={loss_a.item()!r} new={loss_b.item()!r}"

    grads_a, grads_b = _grads(net_a), _grads(net_b)
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        err = (grads_a[name] - grads_b[name]).abs().max().item()
        assert torch.allclose(grads_a[name], grads_b[name], atol=_GRAD_TOL, rtol=0), f"grad mismatch at {name}: max_abs_err={err:.3e}"
