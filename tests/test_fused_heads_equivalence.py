"""WSEL-18 equivalence test (`docs/plans/capacity_programme/width.md` lines 1988-2007).

`SharedTrunkPerWidthHeadNet`'s per-head bookkeeping under the ALL schedule (`w_max` separate
`mean_heads` tensors, each paying its own forward/backward/optimizer dispatch) is the WSEL-14 cost
probe's measured premium, not arithmetic. `fused_heads=True` (constructor flag, default OFF) fuses
the heads into ONE lower-triangular-masked `(w_max, w_max)` weight + `(w_max,)` bias, vectorising
`all_widths_forward` into a single matmul. Under ALL this is claimed MATHEMATICALLY EXACT; this file
is the proof, covering every clause of the spec's point (2):

  - outputs equal per-head at float tolerance on fixed seeds (`forward_width`, `all_widths_forward`,
    `sampled_widths_forward`, including repeated widths);
  - gradients equal per-head at float tolerance on fixed seeds, under the SAME ALL-schedule total
    loss `_train_kdropout_to_convergence` actually backpropagates;
  - masked entries (the upper triangle, `col > row`) remain EXACTLY zero after REAL Adam steps --
    pinned by assertion against a net that has actually been trained, not assumed from the
    architecture alone;
  - per-width best-weight snapshots slice rows and round-trip: a fused row, dropped into a
    standalone `Linear(w_max, 1)`, reproduces `forward_width` exactly;
  - prove-it-fails: an otherwise-identical fused net whose forward path skips the mask (the naive
    vectorisation this task exists to avoid shipping) does NOT keep the dead region at zero under
    training -- proving `_fused_weight()`'s re-mask is load-bearing, not a no-op;
  - the hard error: `fused_heads=True` is refused outside `arch=shared_trunk` + `schedule=all`.

Equivalence-by-construction: `SharedTrunkPerWidthHeadNet.__init__` builds `fused_heads=True` and
`fused_heads=False` nets via the SAME per-row RNG draw order (trunk first, then `w_max` independent
`Linear(w_max, 1)` constructions) -- so two nets built under the same `torch.manual_seed` have
IDENTICAL trunks and IDENTICAL live (unmasked) head weights/biases; only each row's DEAD tail
(columns >= that row's own width, read by neither mode) differs, since the fused tail is hard-zeroed
at init instead of left at its unused random draw. `_paired_nets` below is that construction.

Precision: forward-only comparisons at production float32 use an EXACT (`diff == 0.0`) bar for
`forward_width`/`sampled_widths_forward` -- both sides sum the SAME `w_max` terms with the SAME
`w_max - k` of them contributing an exact `0.0` (IEEE-754's addition identity, order-independent),
measured bit-identical for this shape/backend. `all_widths_forward` reassociates through a batched
GEMM instead of `w_max` measured-independent dot products, so it gets a tolerance
(`test_nested_width_single_trunk.py`'s own precedent for this exact class of float32 non-
associativity noise: measured max abs diff ~6e-8 here, `_FWD_TOL=1e-5` is ~150x looser and still
tight enough to catch a real algorithmic bug). The gradient-equality test runs in float64 (same
precedent) for a tight bit-for-bit bar; the masked-entries-stay-zero property is dtype-independent
(0 gradient -> 0 Adam update is EXACT in either precision, not merely small) so that test runs at
production float32 with production's own optimizer (`cwe.LR`).

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m pytest tests/test_fused_heads_equivalence.py -q
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "automl_package", "examples"))
import converged_width_experiment as cwe
import kdropout_converged_width_experiment as kce
import nested_width_net as nwn

_SEED = 0
_FWD_TOL = 1e-5  # forward, float32 (see module docstring's precision note)
_GRAD_TOL = 1e-10  # gradients, float64 (test_nested_width_single_trunk.py's own convention)

# Forward/gradient equivalence: the canonical shape every real width cell runs at.
_CANONICAL_W_MAX = 12
_CANONICAL_N = 1500

# Training-based tests (masked-zero-after-steps, prove-it-fails): small and cheap, only the masking
# mechanism is under test, not the architecture's fit quality.
_TRAIN_W_MAX = 6
_TRAIN_N = 64
_N_STEPS = 200
_PROVE_IT_FAILS_DRIFT_FLOOR = 1e-6  # dead-region weight must clear this to count as "moved" (prove-it-fails)


def _paired_nets(w_max: int, seed: int, *, double: bool = False) -> tuple[nwn.SharedTrunkPerWidthHeadNet, nwn.SharedTrunkPerWidthHeadNet]:
    """`(unfused, fused)` `SharedTrunkPerWidthHeadNet` pair built under the SAME seed -- see module docstring."""
    torch.manual_seed(seed)
    unfused = nwn.SharedTrunkPerWidthHeadNet(w_max=w_max, fused_heads=False)
    torch.manual_seed(seed)
    fused = nwn.SharedTrunkPerWidthHeadNet(w_max=w_max, fused_heads=True)
    if double:
        unfused, fused = unfused.double(), fused.double()
    return unfused, fused


@pytest.mark.parametrize("k", list(range(1, _CANONICAL_W_MAX + 1)))
def test_forward_width_outputs_match_unfused(k: int) -> None:
    """Outputs equal per-head, every width, float32, EXACT (see module docstring's precision note)."""
    unfused, fused = _paired_nets(_CANONICAL_W_MAX, _SEED)
    x = torch.randn(64, 1)
    mean_u, logvar_u = unfused.forward_width(x, k)
    mean_f, logvar_f = fused.forward_width(x, k)
    diff = (mean_u - mean_f).abs().max().item()
    assert diff == 0.0, f"forward_width mismatch at k={k}: max_abs_diff={diff:.6e}"
    assert torch.equal(logvar_u, logvar_f)  # both are the dummy zero tensor


def test_all_widths_forward_matches_unfused() -> None:
    """Outputs equal per-head, whole `(N, w_max)` table, float32, tolerance (batched-GEMM reassociation)."""
    unfused, fused = _paired_nets(_CANONICAL_W_MAX, _SEED)
    x = torch.randn(64, 1)
    mean_u, logvar_u = unfused.all_widths_forward(x)
    mean_f, logvar_f = fused.all_widths_forward(x)
    diff = (mean_u - mean_f).abs().max().item()
    assert diff < _FWD_TOL, f"all_widths_forward mismatch: max_abs_diff={diff:.6e} (tol={_FWD_TOL:.0e})"
    assert torch.equal(logvar_u, logvar_f)


def test_sampled_widths_forward_matches_unfused_including_repeats() -> None:
    """Outputs equal per-head for a real sandwich-shaped draw WITH a repeated width, float32, tolerance.

    Gathers multiple rows at once (`h @ weight_rows.t()`, a batched GEMM), same reassociation-vs-
    `forward_width`'s-single-row-dot-product story as `all_widths_forward` -- see module docstring.
    """
    unfused, fused = _paired_nets(_CANONICAL_W_MAX, _SEED)
    x = torch.randn(64, 1)
    widths = [1, _CANONICAL_W_MAX, 3, 7, 7]  # repeat -- sampled_widths_forward must preserve order/repeats
    mean_u, _ = unfused.sampled_widths_forward(x, widths)
    mean_f, _ = fused.sampled_widths_forward(x, widths)
    diff = (mean_u - mean_f).abs().max().item()
    assert diff < _FWD_TOL, f"sampled_widths_forward mismatch: max_abs_diff={diff:.6e} (tol={_FWD_TOL:.0e})"


def test_gradients_match_unfused_under_all_schedule_loss() -> None:
    """Gradients equal per-head, under the SAME ALL-schedule total loss production backpropagates.

    float64 (see module docstring): the live (col <= k-1) slice of each fused row's gradient must
    equal the corresponding unfused head's gradient, and the DEAD region's gradient must be exactly
    zero on BOTH sides (unfused: h_masked zeroes it; fused: the weight mask zeroes it).
    """
    unfused, fused = _paired_nets(_CANONICAL_W_MAX, _SEED, double=True)
    x = torch.randn(_CANONICAL_N, 1, dtype=torch.float64)
    y = torch.randn(_CANONICAL_N, dtype=torch.float64)
    widths_all = list(range(1, _CANONICAL_W_MAX + 1))

    loss_u = kce._sampled_widths_total_loss(kce.LossType.MSE, unfused, widths_all, x, y)
    loss_u.backward()
    loss_f = kce._sampled_widths_total_loss(kce.LossType.MSE, fused, widths_all, x, y)
    loss_f.backward()

    assert torch.allclose(unfused.trunk.weight.grad, fused.trunk.weight.grad, atol=_GRAD_TOL, rtol=0)
    assert torch.allclose(unfused.trunk.bias.grad, fused.trunk.bias.grad, atol=_GRAD_TOL, rtol=0)

    for k in range(1, _CANONICAL_W_MAX + 1):
        head_grad = unfused.mean_heads[k - 1].weight.grad[0]  # (w_max,)
        fused_row_grad = fused.fused_mean_weight.grad[k - 1]  # (w_max,)
        live = slice(0, k)
        dead = slice(k, _CANONICAL_W_MAX)
        err = (head_grad[live] - fused_row_grad[live]).abs().max().item()
        assert torch.allclose(head_grad[live], fused_row_grad[live], atol=_GRAD_TOL, rtol=0), f"live weight-grad mismatch at k={k}: max_abs_err={err:.3e}"
        if k < _CANONICAL_W_MAX:  # dead region is empty at k == w_max
            assert head_grad[dead].abs().max().item() == 0.0, f"unfused dead-region grad not exactly zero at k={k}"
            assert fused_row_grad[dead].abs().max().item() == 0.0, f"fused dead-region grad not exactly zero at k={k}"

        bias_err = (unfused.mean_heads[k - 1].bias.grad[0] - fused.fused_mean_bias.grad[k - 1]).abs().item()
        assert bias_err < _GRAD_TOL, f"bias-grad mismatch at k={k}: abs_err={bias_err:.3e}"


def test_masked_entries_exactly_zero_after_real_optimizer_steps() -> None:
    """(2) 'masked entries remain EXACTLY zero after real optimizer steps' -- pinned, not assumed.

    Runs `_N_STEPS` REAL Adam steps (production's own optimizer, `cwe.LR`) on the ALL-schedule total
    loss, then reads BOTH the raw parameter's masked (upper-triangle) region -- must never move off
    its exactly-zero init, since `_fused_weight()`'s mask multiply makes the chain-rule gradient into
    a masked entry exactly `mask_value * upstream = 0` on every step -- AND the always-remasked
    `_fused_weight()` view (structurally zero on every call; checked too so a future refactor that
    drops the remask is caught here).
    """
    torch.manual_seed(_SEED)
    net = nwn.SharedTrunkPerWidthHeadNet(w_max=_TRAIN_W_MAX, fused_heads=True)
    opt = torch.optim.Adam(net.parameters(), lr=cwe.LR)
    gen = torch.Generator().manual_seed(_SEED)
    x = torch.randn(_TRAIN_N, 1, generator=gen)
    y = torch.randn(_TRAIN_N, generator=gen)
    widths_all = list(range(1, _TRAIN_W_MAX + 1))

    for _ in range(_N_STEPS):
        opt.zero_grad()
        loss = kce._sampled_widths_total_loss(kce.LossType.MSE, net, widths_all, x, y)
        loss.backward()
        opt.step()

    dead = net._fused_head_mask == 0
    raw_dead_max = net.fused_mean_weight.detach()[dead].abs().max().item()
    eff_dead_max = net._fused_weight().detach()[dead].abs().max().item()
    assert raw_dead_max == 0.0, f"raw fused_mean_weight's masked region moved off zero after {_N_STEPS} real Adam steps: max_abs={raw_dead_max!r}"
    assert eff_dead_max == 0.0, f"_fused_weight()'s masked region is not exactly zero after training: max_abs={eff_dead_max!r}"


def test_prove_it_fails_without_the_mask() -> None:
    """Prove-it-fails: an otherwise-identical fused readout that skips the mask (the naive vectorised
    matmul this task exists to avoid shipping) does NOT keep the dead region at zero under training --
    proving `_fused_weight()`'s re-mask is load-bearing, not a no-op. If this test does NOT fail, the
    zero-after-training assertion above would be meaningless (nothing would distinguish "the mask
    worked" from "the dead region was never going to move regardless").
    """
    torch.manual_seed(_SEED)
    net = nwn.SharedTrunkPerWidthHeadNet(w_max=_TRAIN_W_MAX, fused_heads=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(_SEED)
    x = torch.randn(_TRAIN_N, 1, generator=gen)
    y = torch.randn(_TRAIN_N, generator=gen)

    for _ in range(_N_STEPS):
        opt.zero_grad()
        h = net.hidden(x)
        mean_all_unmasked = h @ net.fused_mean_weight.t() + net.fused_mean_bias  # NO mask -- the bug this task must not ship.
        loss = ((mean_all_unmasked - y.unsqueeze(1)) ** 2).mean()
        loss.backward()
        opt.step()

    dead = net._fused_head_mask == 0
    dead_after = net.fused_mean_weight.detach()[dead].abs().max().item()
    assert dead_after > _PROVE_IT_FAILS_DRIFT_FLOOR, (
        f"expected the UNMASKED fused matmul to move the dead-region weights off zero under training (max_abs={dead_after!r}) "
        "-- if it stayed at 0, this test is not discriminating anything"
    )


def test_per_width_row_slice_round_trips_to_equivalent_head() -> None:
    """(2) 'per-width best-weight snapshots slice rows and round-trip': slicing fused row `k-1` out and
    dropping it into a standalone `Linear(w_max, 1)` reproduces `forward_width(x, k)` exactly -- the
    shape a per-width best-weight snapshot would need to reconstruct off a fused net.
    """
    torch.manual_seed(_SEED)
    net = nwn.SharedTrunkPerWidthHeadNet(w_max=_CANONICAL_W_MAX, fused_heads=True)
    x = torch.randn(64, 1)
    for k in range(1, _CANONICAL_W_MAX + 1):
        head = nn.Linear(_CANONICAL_W_MAX, 1)
        with torch.no_grad():
            head.weight.copy_(net._fused_weight()[k - 1].unsqueeze(0))
            head.bias.copy_(net.fused_mean_bias[k - 1].unsqueeze(0))
        h = net.hidden(x)
        mean_reconstructed = head(h)
        mean_direct, _ = net.forward_width(x, k)
        diff = (mean_reconstructed - mean_direct).abs().max().item()
        assert diff == 0.0, f"row-slice round-trip mismatch at k={k}: max_abs_diff={diff:.6e}"


def test_fused_heads_refused_for_non_all_schedule() -> None:
    """(3) hard error: `fused_heads=True` + any schedule other than ALL."""
    with pytest.raises(ValueError, match="requires --schedule"):
        kce._validate_fused_heads(kce.Arch.SHARED_TRUNK, nwn.WidthSchedule.SANDWICH, fused_heads=True)
    with pytest.raises(ValueError, match="requires --schedule"):
        kce._validate_fused_heads(kce.Arch.SHARED_TRUNK, nwn.WidthSchedule.UNIFORM, fused_heads=True)


def test_fused_heads_refused_for_non_shared_trunk_arch() -> None:
    """(3) hard error: `fused_heads=True` on any architecture other than `SHARED_TRUNK`."""
    for arch in (kce.Arch.NESTED, kce.Arch.INDEPENDENT, kce.Arch.AFFINE_SEAM):
        with pytest.raises(ValueError, match="requires --arch"):
            kce._validate_fused_heads(arch, nwn.WidthSchedule.ALL, fused_heads=True)


def test_fused_heads_false_never_raises() -> None:
    """`fused_heads=False` is a no-op regardless of arch/schedule -- existing paths stay untouched."""
    for arch in kce.Arch:
        for schedule in nwn.WidthSchedule:
            kce._validate_fused_heads(arch, schedule, fused_heads=False)  # must not raise
