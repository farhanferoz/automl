"""WSEL-19 §2b equivalence test -- the input-dimension generalization of `SharedTrunkPerWidthHeadNet`
and `IndependentWidthNet` (`docs/plans/capacity_programme/shared/wsel19-toy-design.md` §2b, F1).

Both classes' trunks moved from a hardcoded `Linear(1, ...)` to a constructor-time `in_dim`
parameter (default 1). This file is the equivalence proof the amended spec requires BEFORE any
`d > 1` grid cell may run (§2b(a)): default (`in_dim` omitted) and explicit `in_dim=1` must build
BYTE-IDENTICAL nets under the same seed -- same parameter shapes, same initialization draws, same
forward outputs -- and `in_dim=8` must produce the documented `(N, 8) -> (N, w_max)` shape with the
WSEL-18 fused-mode masked-region invariant still intact.

Covers:
  - `test_*_default_equals_explicit_in_dim_1_*`: parameter-for-parameter and forward-output
    identity between the pre-generalization constructor call and the new one at its default, for
    both classes.
  - `test_*_in_dim_8_shape`: the class accepts `in_dim=8` and maps `(N, 8) -> (N, w_max)`.
  - `test_fused_heads_mask_still_exact_at_in_dim_8`: WSEL-18's fused-mode masked-(dead)-region-zero
    invariant survives at `in_dim > 1` (composes cleanly with the §2b change).
  - `test_prove_it_fails_*`: a trunk rebuilt at the OLD hardcoded `in_dim=1` (the bug this task must
    not ship) is shown to actually break on an 8-feature input, then the real trunk is restored and
    shown to work -- proving the shape tests above discriminate the real wiring, not a tautology.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_width_indim_equivalence.py -q
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from automl_package.models.flexnn.width.architectures import IndependentWidthNet, SharedTrunkPerWidthHeadNet

_SEED = 0
_W_MAX = 6
_N = 37
_D = 8
_FUSED_FWD_TOL = 1e-5  # batched-GEMM (all_widths_forward) vs per-row dot product (forward_width) reassociation noise -- test_fused_heads_equivalence.py's own precedent.


def _shared_trunk_pair(seed: int, w_max: int) -> tuple[SharedTrunkPerWidthHeadNet, SharedTrunkPerWidthHeadNet]:
    """`(default, explicit in_dim=1)` `SharedTrunkPerWidthHeadNet` pair built under the SAME seed."""
    torch.manual_seed(seed)
    default_net = SharedTrunkPerWidthHeadNet(w_max=w_max)
    torch.manual_seed(seed)
    explicit_net = SharedTrunkPerWidthHeadNet(w_max=w_max, in_dim=1)
    return default_net, explicit_net


def _independent_pair(seed: int, w_max: int) -> tuple[IndependentWidthNet, IndependentWidthNet]:
    """`(default, explicit in_dim=1)` `IndependentWidthNet` pair built under the SAME seed."""
    torch.manual_seed(seed)
    default_net = IndependentWidthNet(w_max=w_max)
    torch.manual_seed(seed)
    explicit_net = IndependentWidthNet(w_max=w_max, in_dim=1)
    return default_net, explicit_net


def test_shared_trunk_default_equals_explicit_in_dim_1_parameters() -> None:
    """Same seed -> IDENTICAL trunk/head parameter tensors, default vs explicit `in_dim=1`."""
    default_net, explicit_net = _shared_trunk_pair(_SEED, _W_MAX)
    assert torch.equal(default_net.trunk.weight, explicit_net.trunk.weight)
    assert torch.equal(default_net.trunk.bias, explicit_net.trunk.bias)
    for k in range(_W_MAX):
        assert torch.equal(default_net.mean_heads[k].weight, explicit_net.mean_heads[k].weight)
        assert torch.equal(default_net.mean_heads[k].bias, explicit_net.mean_heads[k].bias)


def test_shared_trunk_default_equals_explicit_in_dim_1_forward() -> None:
    """Same seed, same input -> IDENTICAL forward outputs, every width, default vs explicit `in_dim=1`."""
    default_net, explicit_net = _shared_trunk_pair(_SEED, _W_MAX)
    x = torch.randn(_N, 1)
    for k in range(1, _W_MAX + 1):
        mean_d, logvar_d = default_net.forward_width(x, k)
        mean_e, logvar_e = explicit_net.forward_width(x, k)
        assert torch.equal(mean_d, mean_e)
        assert torch.equal(logvar_d, logvar_e)
    mean_all_d, logvar_all_d = default_net.all_widths_forward(x)
    mean_all_e, logvar_all_e = explicit_net.all_widths_forward(x)
    assert torch.equal(mean_all_d, mean_all_e)
    assert torch.equal(logvar_all_d, logvar_all_e)


def test_independent_default_equals_explicit_in_dim_1_parameters() -> None:
    """Same seed -> IDENTICAL per-subnet parameter tensors, default vs explicit `in_dim=1`."""
    default_net, explicit_net = _independent_pair(_SEED, _W_MAX)
    for k in range(_W_MAX):
        sub_d, sub_e = default_net.subnets[k], explicit_net.subnets[k]
        assert torch.equal(sub_d["trunk"].weight, sub_e["trunk"].weight)
        assert torch.equal(sub_d["trunk"].bias, sub_e["trunk"].bias)
        assert torch.equal(sub_d["mean_head"].weight, sub_e["mean_head"].weight)
        assert torch.equal(sub_d["logvar_head"].weight, sub_e["logvar_head"].weight)


def test_independent_default_equals_explicit_in_dim_1_forward() -> None:
    """Same seed, same input -> IDENTICAL forward outputs, every width, default vs explicit `in_dim=1`."""
    default_net, explicit_net = _independent_pair(_SEED, _W_MAX)
    x = torch.randn(_N, 1)
    mean_all_d, logvar_all_d = default_net.all_widths_forward(x)
    mean_all_e, logvar_all_e = explicit_net.all_widths_forward(x)
    assert torch.equal(mean_all_d, mean_all_e)
    assert torch.equal(logvar_all_d, logvar_all_e)


def test_shared_trunk_in_dim_8_shape() -> None:
    """The constructor accepts `in_dim=8`; forward maps `(N, 8) -> (N, w_max)` per-width outputs."""
    torch.manual_seed(_SEED)
    net = SharedTrunkPerWidthHeadNet(w_max=_W_MAX, in_dim=_D)
    x = torch.randn(_N, _D)
    mean_k, _log_var_k = net.forward_width(x, 3)
    assert mean_k.shape == (_N, 1)
    mean_all, _log_var_all = net.all_widths_forward(x)
    assert mean_all.shape == (_N, _W_MAX)


def test_independent_in_dim_8_shape() -> None:
    """The constructor accepts `in_dim=8`; forward maps `(N, 8) -> (N, w_max)` per-width outputs."""
    torch.manual_seed(_SEED)
    net = IndependentWidthNet(w_max=_W_MAX, in_dim=_D)
    x = torch.randn(_N, _D)
    mean_k, _log_var_k = net.forward_width(x, 3)
    assert mean_k.shape == (_N, 1)
    mean_all, _log_var_all = net.all_widths_forward(x)
    assert mean_all.shape == (_N, _W_MAX)


def test_fused_heads_mask_still_exact_at_in_dim_8() -> None:
    """WSEL-18's fused-mode masked (dead) region stays EXACTLY zero when composed with `in_dim > 1`."""
    torch.manual_seed(_SEED)
    net = SharedTrunkPerWidthHeadNet(w_max=_W_MAX, fused_heads=True, in_dim=_D)
    x = torch.randn(_N, _D)
    mean_all, _log_var_all = net.all_widths_forward(x)
    assert mean_all.shape == (_N, _W_MAX)
    for k in range(1, _W_MAX + 1):
        mean_k, _log_var_k = net.forward_width(x, k)
        assert torch.allclose(mean_all[:, k - 1 : k], mean_k, atol=_FUSED_FWD_TOL, rtol=0.0)
    dead = net._fused_head_mask == 0
    assert net._fused_weight().detach()[dead].abs().max().item() == 0.0


def test_prove_it_fails_wrong_trunk_wiring_breaks_shared_trunk_in_dim_8() -> None:
    """Prove-it-fails: a trunk rebuilt at the OLD hardcoded `in_dim=1` (the bug this task must not
    ship, as if `__init__` had never wired `in_dim` through) does NOT accept an 8-feature input --
    then the real trunk is restored and shown to work, so the shape test above discriminates the
    correct wiring from the regression, not a tautology.
    """
    torch.manual_seed(_SEED)
    net = SharedTrunkPerWidthHeadNet(w_max=_W_MAX, in_dim=_D)
    x = torch.randn(_N, _D)
    correct_trunk = net.trunk

    net.trunk = nn.Linear(1, net.w_max)  # the regression: in_dim ignored, hardcoded back to 1.
    with pytest.raises(RuntimeError):
        net.forward_width(x, 3)

    net.trunk = correct_trunk  # restore -- the real wiring works.
    mean_k, _log_var_k = net.forward_width(x, 3)
    assert mean_k.shape == (_N, 1)


def test_prove_it_fails_wrong_trunk_wiring_breaks_independent_in_dim_8() -> None:
    """Prove-it-fails, `IndependentWidthNet`: one sub-net's trunk rebuilt at the OLD hardcoded
    `in_dim=1` does NOT accept an 8-feature input; restoring it works again.
    """
    torch.manual_seed(_SEED)
    net = IndependentWidthNet(w_max=_W_MAX, in_dim=_D)
    x = torch.randn(_N, _D)
    k = 3
    sub = net.subnets[k - 1]
    correct_trunk = sub["trunk"]

    sub["trunk"] = nn.Linear(1, k)  # the regression: in_dim ignored, hardcoded back to 1.
    with pytest.raises(RuntimeError):
        net.forward_width(x, k)

    sub["trunk"] = correct_trunk  # restore -- the real wiring works.
    mean_k, _log_var_k = net.forward_width(x, k)
    assert mean_k.shape == (_N, 1)
