"""Scratch equivalence probe for WSEL-12 review (read-only against the repo).

Checks the landed test suite (tests/test_nested_width_single_trunk.py) does NOT cover:
  (1) widths lists with REPEATS (UNIFORM schedule draws WITH replacement -- torch.randint --
      so the same k can appear twice+ in one step's `widths` list).
  (2) the ALL schedule's full widths=range(1, w_max+1) list, all three shared-trunk archs.

For both, compares:
  (a) old per-k loop (`kce._width_loss` summed one k at a time -- the pre-WSEL-12 accumulation)
  (b) new `kce._sampled_widths_total_loss` (single sampled_widths_forward call)
at float32 (production dtype, forward only -- bit-identical is the bar) and float64 (gradient check).
"""
from __future__ import annotations

import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join("/home/ff235/dev/MLResearch/automl", "automl_package", "examples"))
import kdropout_converged_width_experiment as kce  # noqa: E402
import nested_width_net as nwn  # noqa: E402

ARCHS = (nwn.NestedWidthNet, nwn.SharedTrunkPerWidthHeadNet, nwn.SharedReadoutPerWidthAffineNet)
W_MAX = 12
N = 1500
_SMALL_W_MAX = 6
_LOSS_TOL = 1e-12


def old_loop_loss(loss, net, widths, x, y):
    total = torch.zeros((), dtype=x.dtype)
    for k in widths:
        total = total + kce._width_loss(loss, net, k, x, y)
    return total


def check_forward_bit_identical(widths, label):
    print(f"\n=== forward bit-identical check: {label} (widths={widths}) ===")
    for arch in ARCHS:
        for loss in kce.LossType:
            for seed in (0, 1, 2):
                torch.manual_seed(seed)
                net = arch(w_max=W_MAX)
                x = torch.randn(N, 1)
                y = torch.randn(N, 1)
                old = old_loop_loss(loss, net, widths, x, y)
                new = kce._sampled_widths_total_loss(loss, net, widths, x, y)
                diff = (old - new).abs().item()
                status = "OK" if diff == 0.0 else "MISMATCH"
                if status != "OK":
                    print(f"  {arch.__name__}/{loss.value}/seed={seed}: diff={diff:.6e}  <-- {status}")
        print(f"  {arch.__name__}: forward loss checked for all losses/seeds")


def check_grad_agreement_float64(widths, label, atol=1e-10):
    print(f"\n=== gradient agreement (float64) check: {label} (widths={widths}) ===")
    for arch in ARCHS:
        torch.manual_seed(0)
        net = arch(w_max=_SMALL_W_MAX if max(widths) <= _SMALL_W_MAX else W_MAX).double()
        x = torch.randn(64, 1, dtype=torch.float64)
        y = torch.randn(64, 1, dtype=torch.float64)
        net_a = copy.deepcopy(net)
        net_b = copy.deepcopy(net)
        loss_a = old_loop_loss(kce.LossType.MSE, net_a, widths, x, y.squeeze(1))
        loss_a.backward()
        loss_b = kce._sampled_widths_total_loss(kce.LossType.MSE, net_b, widths, x, y.squeeze(1))
        loss_b.backward()
        loss_diff = (loss_a - loss_b).abs().item()
        max_grad_err = 0.0
        for (na, pa), (nb, pb) in zip(net_a.named_parameters(), net_b.named_parameters(), strict=True):
            assert na == nb
            ga = pa.grad if pa.grad is not None else torch.zeros_like(pa)
            gb = pb.grad if pb.grad is not None else torch.zeros_like(pb)
            err = (ga - gb).abs().max().item()
            max_grad_err = max(max_grad_err, err)
        ok = loss_diff <= _LOSS_TOL and max_grad_err <= atol
        print(f"  {arch.__name__}: loss_diff={loss_diff:.3e} max_grad_err={max_grad_err:.3e}  {'OK' if ok else 'FAIL(' + str(atol) + ')'}")


if __name__ == "__main__":
    # (1) REPEATS -- e.g. UNIFORM schedule draw with replacement can produce e.g. [3, 3, 5, 12].
    repeated_widths = [3, 3, 5, 12, 1, 1]
    check_forward_bit_identical(repeated_widths, "repeats (uniform-with-replacement-like)")
    check_grad_agreement_float64([3, 3, 5, 6], "repeats, small w_max=6")

    # (2) ALL schedule -- every width 1..w_max, once each, deterministic.
    all_widths = list(range(1, W_MAX + 1))
    check_forward_bit_identical(all_widths, "ALL schedule (every width)")
    check_grad_agreement_float64(list(range(1, 7)), "ALL schedule, small w_max=6")

    # (3) UNIFORM-like draw, moderate length, with heavy repetition
    heavy_repeat = [5] * 8 + [1, 12]
    check_forward_bit_identical(heavy_repeat, "heavy repeats of one width")

    print("\nDONE")
