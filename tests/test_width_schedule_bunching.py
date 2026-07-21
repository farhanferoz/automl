"""WSEL-14 Step 2 — pins the bunch-size-1 optimiser footgun (`docs/plans/capacity_programme/width.md`).

Under bunch size 1 (only one width trained per step -- `--schedule uniform --uniform-draw-n 1`), a
width whose OWN per-width head (`SharedTrunkPerWidthHeadNet.mean_heads`) is not among the widths
sampled that step must get `grad is None` after `backward()`, not a zero tensor -- and its values must
be UNCHANGED after `opt.step()`. This is the mechanism that makes bunch size 1 gradient-correct (no
phantom decay/momentum drift on heads no batch asked to touch), verified against plain Adam at
`automl_package/examples/nested_width_net.py:271` and PyTorch 2.10's `zero_grad(set_to_none=True)`
default (`torch.optim.Optimizer.zero_grad`'s own default).

**Prove-it-fails companion**: with `zero_grad(set_to_none=False)`, an unselected head's `.grad` is left
as an explicit ZERO tensor (not None) instead of being cleared. A zero gradient with weight decay is
NOT a no-op for `torch.optim.Adam` -- Adam's L2-style weight decay adds `weight_decay * param` to the
gradient BEFORE the moment-estimate bookkeeping, so an ostensibly-untouched head shrinks toward zero
even though no batch selected it. This test demonstrates the failure directly (not merely asserted) so
a future change to `zero_grad`'s call site or a stray `weight_decay>0` regresses loudly.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m pytest tests/test_width_schedule_bunching.py -q
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "automl_package", "examples"))
import kdropout_converged_width_experiment as kce
import nested_width_net as nwn


def _bunch1_net_and_batch() -> tuple[nwn.SharedTrunkPerWidthHeadNet, torch.Tensor, torch.Tensor]:
    """A tiny `SharedTrunkPerWidthHeadNet` (per-width heads -- the class the footgun bites) plus a batch."""
    torch.manual_seed(0)
    net = nwn.SharedTrunkPerWidthHeadNet(w_max=3)
    x = torch.randn(16, 1)
    y = torch.randn(16)
    return net, x, y


def test_bunch_size_one_unselected_head_grad_is_none_and_untouched():
    """Bunch size 1, `set_to_none=True` (the real default): unselected heads get grad=None and don't move."""
    net, x, y = _bunch1_net_and_batch()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)  # plain Adam, no weight decay -- matches nested_width_net.py:271

    unselected_head = net.mean_heads[2]  # width 3's OWN head; only width 2 is sampled this step
    before = unselected_head.weight.detach().clone()

    opt.zero_grad(set_to_none=True)
    total_loss = kce._sampled_widths_total_loss(kce.LossType.MSE, net, [2], x, y)
    total_loss.backward()
    assert unselected_head.weight.grad is None, "an unselected width's head must get grad=None, not a zero tensor, after backward()"

    opt.step()
    after = unselected_head.weight.detach().clone()
    assert torch.equal(before, after), "an unselected width's head must be unchanged by opt.step() when its grad is None"


def test_bunch_size_one_footgun_prove_it_fails_with_set_to_none_false_and_weight_decay():
    """Prove-it-fails: `set_to_none=False` + weight decay DOES shrink an untouched head -- the footgun is real."""
    net, x, y = _bunch1_net_and_batch()
    opt = torch.optim.Adam(net.parameters(), lr=1e-1, weight_decay=1e-1)
    unselected_head = net.mean_heads[2]

    # Prime: one step that trains EVERY width, so mean_heads[2] gets a real (non-None) .grad tensor and
    # Adam optimiser state. Without this, `zero_grad(set_to_none=False)` has nothing to zero -- a `.grad`
    # that was never created stays None regardless of the flag, which would hide the footgun entirely.
    opt.zero_grad(set_to_none=True)
    kce._sampled_widths_total_loss(kce.LossType.MSE, net, [1, 2, 3], x, y).backward()
    opt.step()

    before = unselected_head.weight.detach().clone()

    # The footgun step: only width 2 is sampled this time, but `set_to_none=False` leaves
    # mean_heads[2]'s `.grad` as an explicit ZERO tensor (not None) instead of clearing it -- Adam's L2
    # weight decay then adds `weight_decay * param` to that "zero" gradient before its moment-estimate
    # bookkeeping, so the step below is NOT a no-op even though width 3 was not sampled.
    opt.zero_grad(set_to_none=False)
    total_loss = kce._sampled_widths_total_loss(kce.LossType.MSE, net, [2], x, y)
    total_loss.backward()
    assert unselected_head.weight.grad is not None
    assert torch.equal(unselected_head.weight.grad, torch.zeros_like(unselected_head.weight))

    opt.step()
    after = unselected_head.weight.detach().clone()
    assert not torch.equal(before, after), (
        "expected the footgun to reproduce: set_to_none=False + weight_decay>0 should shrink a head no batch asked to change"
    )


def test_results_dir_flag_absent_resolves_to_the_legacy_directory():
    """`--results-dir` absent must resolve to this module's historic output directory, unchanged.

    Other drivers (`width_wsel13.py`, `width_wsel15.py`) import this module and rely on that default
    never silently relocating their outputs -- this pins it so a later "tidy" of the default can't.
    """
    assert kce._resolve_results_dir(None) == kce.RESULTS_DIR
    assert kce.RESULTS_DIR.endswith(os.path.join("capacity_ladder_results", "W_KDROPOUT_CONVERGED"))


def test_results_dir_flag_override_is_used_when_given():
    """`--results-dir`, when given, is used verbatim (not merged with, or relative to, the legacy default)."""
    override = "/tmp/wsel14_results_dir_test_override"
    assert kce._resolve_results_dir(override) == override
