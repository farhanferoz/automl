"""WSEL-17 Step 1 -- proves the §3.9 duplicate-pair finding by measurement, not by taste.

`MatryoshkaWidthNet` (`automl_package/examples/matryoshka_width_net.py`, pre-cleanup: heads
`Linear(k -> 1)` reading the raw prefix `h[:, :k]`) and the certified `SharedTrunkPerWidthHeadNet`
(`automl_package/models/architectures/nested_width_net.py`, heads `Linear(w_max -> 1)` reading a
MASKED hidden vector, columns `>= k` zeroed) are the SAME design: a dedicated per-width readout on
the shared trunk's hidden vector. They differ in ONLY one thing -- the certified head carries
`w_max - k` extra weight columns that can never influence a width-`k` output, because the input
those columns would multiply is already zero (`docs/plans/capacity_programme/width.md` ~2292-2301).

This file is that proof:
  - `test_masked_head_equals_dedicated_prefix_head`: for every `k`, the certified head's width-`k`
    output equals a from-scratch `Linear(k, 1)` reading the UNMASKED prefix `h[:, :k]`, built from
    the same live weight columns (`< k`) and the same bias -- i.e. exactly what `MatryoshkaWidthNet`
    would compute, to `1e-5`, on a fixed seed.
  - `test_zeroing_the_dead_columns_is_a_no_op`: the literal spec wording -- explicitly zero the
    masked head's columns `>= k` in a clone and show `forward_width` is UNCHANGED. This is the
    formal statement of "differ only in nominal parameter count": the dead columns already cannot
    move the output, zeroing them is provably a no-op.
  - `test_perturbing_a_live_column_breaks_the_equivalence` (prove-it-fails): perturbing a LIVE
    column (`< k`, the region both designs actually read) MUST break the match -- otherwise the
    equivalence above would hold trivially regardless of what the live weights are, and would prove
    nothing.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_matryoshka_equivalence.py -q
"""

from __future__ import annotations

import copy

import torch

from automl_package.models.architectures.nested_width_net import SharedTrunkPerWidthHeadNet

_SEED = 0
_W_MAX = 6
_N = 37
_TOL = 1e-5
_PERTURB = 5.0


def _dedicated_prefix_head_output(h: torch.Tensor, weight_live: torch.Tensor, bias: torch.Tensor, k: int) -> torch.Tensor:
    """What `MatryoshkaWidthNet.forward_width(x, k)` computes: `Linear(k, 1)` on the raw prefix `h[:, :k]`.

    Args:
        h: `(N, w_max)` shared-trunk hidden vector, UNMASKED.
        weight_live: `(1, k)` the head's live (`< k`) weight columns.
        bias: `(1,)` the head's bias.
        k: the width.
    """
    return h[:, :k] @ weight_live.t() + bias


def test_masked_head_equals_dedicated_prefix_head() -> None:
    """The certified masked head IS a dedicated `Linear(k, 1)` prefix head, to `1e-5`, every width."""
    torch.manual_seed(_SEED)
    net = SharedTrunkPerWidthHeadNet(w_max=_W_MAX)
    net.eval()
    x = torch.randn(_N, 1)

    with torch.no_grad():
        h = net.hidden(x)
        for k in range(1, _W_MAX + 1):
            head = net.mean_heads[k - 1]
            weight_live = head.weight[:, :k]
            masked_out, _ = net.forward_width(x, k)
            dedicated_out = _dedicated_prefix_head_output(h, weight_live, head.bias, k)
            diff = (masked_out - dedicated_out).abs().max().item()
            assert diff < _TOL, f"width {k}: masked-head vs dedicated-prefix-head mismatch {diff:.3e} (tol={_TOL:.0e})"


def test_zeroing_the_dead_columns_is_a_no_op() -> None:
    """Literal spec wording: explicitly zero the masked head's columns `>= k` and show the output is UNCHANGED."""
    torch.manual_seed(_SEED)
    net = SharedTrunkPerWidthHeadNet(w_max=_W_MAX)
    net.eval()
    x = torch.randn(_N, 1)

    for k in range(1, _W_MAX + 1):
        with torch.no_grad():
            before, _ = net.forward_width(x, k)

        zeroed_net = copy.deepcopy(net)
        with torch.no_grad():
            zeroed_net.mean_heads[k - 1].weight[:, k:] = 0.0
            after, _ = zeroed_net.forward_width(x, k)

        diff = (before - after).abs().max().item()
        assert diff == 0.0, f"width {k}: zeroing dead columns (>= k) changed the output by {diff:.3e} -- they were not already inert"


def test_perturbing_a_live_column_breaks_the_equivalence() -> None:
    """Prove-it-fails: perturbing a LIVE column (< k) must break the masked-vs-dedicated match."""
    torch.manual_seed(_SEED)
    net = SharedTrunkPerWidthHeadNet(w_max=_W_MAX)
    net.eval()
    x = torch.randn(_N, 1)
    k = 4
    head = net.mean_heads[k - 1]

    with torch.no_grad():
        h = net.hidden(x)
        weight_live = head.weight[:, :k].clone()
        bias = head.bias.clone()

        # Perturb a LIVE column (column 0 is < k, so it DOES influence width-k's output) directly on
        # the net, then read the net's own forward_width -- NOT the frozen `weight_live` snapshot.
        head.weight[:, 0] += _PERTURB
        masked_out, _ = net.forward_width(x, k)

    dedicated_out = _dedicated_prefix_head_output(h, weight_live, bias, k)
    diff = (masked_out - dedicated_out).abs().max().item()
    assert diff > _TOL, (
        f"perturbing a live column (diff={diff:.3e}) should have broken the equivalence past tol={_TOL:.0e}, "
        "but it did not -- the comparison above is not discriminating anything"
    )
