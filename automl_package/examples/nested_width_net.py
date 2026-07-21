"""W1 — nested-width probabilistic net: the WIDTH analog of the depth lane's NESTED schedule.

(docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md §2a)

**The four architecture classes moved to `automl_package.models.architectures.nested_width_net`**
(2026-07-21, `docs/plans/capacity_programme/flexnn-package.md` Task FP-2 -- the boundary rule,
MASTER Decision 19). This module re-exports them (below) so every existing `import nested_width_net
as nwn` / `nwn.NestedWidthNet` call site keeps resolving unchanged (the `automl_package/examples/
convergence.py` precedent: move the logic, leave the shim, do not rewrite callers). Everything else
in this file -- the toy generators, the training schedule, the selftest CLI -- stays here, since it
is experiment protocol, not library architecture (FP-2 non-goals).

Depth is fixed at a single hidden layer everywhere in this file; the ONLY capacity axis is the
number of ACTIVE hidden nodes ("width"). Architecture: `Linear(1 -> w_max)` -> activation ->
hidden vector `h in R^{w_max}`; two linear readouts off `h` (`mean_head`, `logvar_head`), giving
a native `(mean, log_var)` Gaussian head — strictly probabilistic, matching
`capacity_ladder_f2._fixed_depth_log_likelihood`'s formula.

Width-k forward: zero `h[k:]` (columns k..w_max-1) before BOTH readouts, so a width-k prediction
depends on the first k hidden nodes only and is a literal PREFIX of width-(k+1) (shared weights,
nested-dropout / slimmable-network structure) -- the node-axis analog of the depth lane's
`hidden_layers_blocks` prefix property (`automl_package/models/flexible_neural_network.py`,
`LayerSelectionMethod.NESTED` in `automl_package/enums.py`). `all_widths_forward` returns every
width's readout from ONE forward pass: because both readouts are LINEAR in `h`, masking nodes
`k..` to zero before a linear layer is exactly a cumulative sum of each node's own (fixed) linear
contribution over the first k nodes -- so the whole `(N, w_max)` table falls out of one hidden
pass plus a `cumsum`, no per-width python loop needed (the width analog of
`NestedStrategy.all_depth_outputs`'s "one forward pass, every depth" trick, layer_selection_
strategies.py:210-227, but exact-and-vectorized here rather than loop-over-blocks since nodes,
unlike depth blocks, don't chain through activations).

`train_nested_width` is the NESTED-width training SCHEDULE (not a learned selector): every
training epoch, each example independently draws its own width `k ~ Uniform{1..w_max}`, is
scored at that drawn width only, and the mean Gaussian NLL over the batch is backpropagated —
mirrors `NestedStrategy.forward`'s per-sample depth draw (layer_selection_strategies.py:252-275)
and `capacity_ladder_h1.py::_train_phase1`'s full-batch draw-per-epoch loop shape. No selector
head exists yet in this file; the selector is `sinc_width_experiment.py`'s phase 2, built on top
of a FROZEN net trained here.

Selftest (`--selftest`, no training, random init only) proves three things about an untrained net:
  (a) prefix invariance — a width-k forward is IDENTICAL whether or not the readout weights of
      nodes >= k are perturbed (those columns are masked to zero before the readout matmul, so
      they cannot affect the result; this is a numerical regression test of the masking, not a
      restatement of the architecture).
  (b) nesting agreement — width-(k+1) minus width-k equals EXACTLY the newly-included node's own
      linear contribution (`h[:, k] * head.weight[0, k]`), i.e. nodes < k contribute identically
      to both widths (the nested-prefix property, mirroring `_flexnn_prefix_selftest.py`'s
      cached-vs-independent-forward comparison for depth).
  (c) the all-widths pass returns finite `(mean, log_var)` of shape `(N, w_max)` for both heads,
      AND agrees with the per-width masked forward at every k (the two code paths -- explicit
      masking, `forward_width`, and the vectorized cumsum, `all_widths_forward` -- must compute
      the same thing).

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/nested_width_net.py --selftest
"""

from __future__ import annotations

import argparse
import enum
import math
import os
import sys

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

from automl_package.models.architectures.nested_width_net import (  # noqa: E402
    W_MAX_DEFAULT,
    IndependentWidthNet,
    NestedWidthNet,
    SharedReadoutPerWidthAffineNet,
    SharedTrunkPerWidthHeadNet,
)

__all__ = [
    "W_MAX_DEFAULT",
    "IndependentWidthNet",
    "NestedWidthNet",
    "SharedReadoutPerWidthAffineNet",
    "SharedTrunkPerWidthHeadNet",
]

_PREFIX_TOL = 1e-6
_NESTING_TOL = 1e-5
_CONSISTENCY_TOL = 1e-5

HETERO_NOISE_SIGMA = 0.05
HETERO_R_DEFAULT = 4.0 * math.pi
HETERO3_NOISY_SIGMA = 0.5  # WP-3 noisy-easy region: 10x the quiet-region sigma (docs/plans/width_mse_2026-07-16 §WP-3).


class Toy(enum.Enum):
    """Which synthetic generator a run uses (`--toy`; closed set, width-MSE program WP-3/WP-4).

    `docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md` §WP-3, §WP-4.
    """

    HETERO = "hetero"  # 2-region easy-linear + hard-sine (`make_hetero`); WP-2/WP-4 ladder toy.
    HETERO3 = "hetero3"  # 3-region: adds a NOISY-easy tail (`make_hetero3`); WP-3 discriminating control.


class WidthSchedule(enum.Enum):
    """Which per-step width-sampling schedule `train_nested_width` uses (closed set, not a string).

    `docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md` §6.
    """

    NESTED = "nested"  # W1: per-example uniform width draw each epoch (default; sinc keeps using this).
    SANDWICH = "sandwich"  # W2 fix: every step ALWAYS trains width=1 and width=w_max (+2 random mid).
    UNIFORM = "uniform"  # W5 ablation: draw N widths uniformly per step, NO always-include-{1, w_max} guarantee.


# The four architecture classes (NestedWidthNet, IndependentWidthNet, SharedTrunkPerWidthHeadNet,
# SharedReadoutPerWidthAffineNet) moved to automl_package.models.architectures.nested_width_net
# (FP-2, see module docstring); imported above and re-exported for every existing `nwn.ClassName`
# call site.


def gaussian_log_likelihood(mean: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-example Gaussian log-likelihood — same formula as `capacity_ladder_f2._fixed_depth_log_likelihood`."""
    variance = torch.exp(log_var)
    return -0.5 * (math.log(2.0 * math.pi) + log_var + (y - mean) ** 2 / variance)


def _width_mse(net: NestedWidthNet | IndependentWidthNet | SharedTrunkPerWidthHeadNet, k: int, x_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """Mean squared error of width-k's MEAN readout only (width-MSE program's `--loss mse`).

    Architecture-agnostic — both `NestedWidthNet` and `IndependentWidthNet` expose `forward_width` —
    the MSE twin of `converged_width_experiment._width_nll`, which trains widths via Gaussian NLL.
    `logvar_head` is not read here, so it is never in this loss's autograd graph and stays untouched
    (`docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md` §0: no variance fitting anywhere).
    """
    mean, _log_var = net.forward_width(x_t, k)
    return ((mean.squeeze(1) - y_t) ** 2).mean()


def make_hetero(n: int, seed: int, r: float = HETERO_R_DEFAULT, sigma: float = HETERO_NOISE_SIGMA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """W2 heterogeneous toy: a flat-easy line spliced to a width-hungry-but-learnable sine (§6 W2 toy).

    `x ~ Uniform(-r, r)`. Easy region (`x < 0`): `y = (0.5/r) * x` (a straight line — width-flat,
    1 node suffices). Hard region (`x >= 0`): `y = 0.5 * sin(x)` (2 native-frequency periods over
    `[0, r]` at the default `r=4*pi`; spreading the periods over a wide range keeps the input-space
    frequency ~1, which a small net can learn — unlike a packed `sin(2*pi*P*x)` on `[0,1]`, which
    is UNLEARNABLE by a small net, the same spectral-bias wall as the tent map, confirmed in
    `scratchpad/hetero_toy_probe.py` for P=2,3,4). Both branches are 0 at `x=0`, so `y`'s noise-free
    signal is continuous there. Gaussian noise, std `sigma`, is added to `y`. Probed
    learnable with a clean per-region width gradient in `scratchpad/hetero_toy_probe_v2.py` (easy
    flat ~1.2-2x the noise floor at every width; hard 52x(w=1) -> 1.3x(w=10)).

    Args:
        n: number of points.
        seed: RNG seed (`np.random.default_rng`).
        r: domain half-width; `x ~ Uniform(-r, r)` (default `4*pi`, per the plan's W2 toy spec).
        sigma: Gaussian noise std, COMMON-MODE across both regions (default `HETERO_NOISE_SIGMA=0.05`).
            The width-MSE program's WP-4 data-size x noise ladder sweeps this on BOTH regions
            uniformly, so the analytic per-region noise floor stays `sigma**2` and the fit-bar formula
            (`docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md` §5) remains computable.

    Returns:
        `(x, y, region)`, each shape `(n,)`, `x`/`y` float32. `region` is an int array: `0` = easy
        (`x < 0`), `1` = hard (`x >= 0`). `x` is NOT standardized here — `x` spans `+-r`, which
        saturates `tanh` at typical init scales, so callers standardize using TRAIN-set statistics
        only and apply the SAME stats to held-out `x` (the pattern `sinc_width_experiment.py`'s
        `_fit_phase1`/`_score_all_widths` already use for the sinc toy).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-r, r, n)
    y_signal = np.where(x < 0, (0.5 / r) * x, 0.5 * np.sin(x))
    y = (y_signal + rng.normal(0.0, sigma, n)).astype(np.float32)
    region = (x >= 0).astype(int)
    return x.astype(np.float32), y, region


def make_hetero3(n: int, seed: int, r: float = HETERO_R_DEFAULT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WP-3 discriminating control: `make_hetero` plus a third NOISY-easy region (the negative control).

    `x ~ Uniform(-r, 2r)` — three equal-length regions of width `r` (default `4*pi`), so per-region
    density matches the 2-region toy when `n` is scaled by 3/2:

      - region `0` (easy-linear, `x in [-r, 0)`): `y = (0.5/r) * x`, noise std `HETERO_NOISE_SIGMA=0.05`.
      - region `1` (hard-sine, `x in [0, r)`): `y = 0.5 * sin(x)`, noise std `HETERO_NOISE_SIGMA=0.05`.
      - region `2` (NOISY-easy-linear, `x in [r, 2r)`): `y = (0.5/r) * (x - r)`, noise std
        `HETERO3_NOISY_SIGMA=0.5` (10x the quiet regions).

    The noise-free signal is continuous at both seams: at `x=0` easy and hard both give `0`; at `x=r`
    the hard sine gives `0.5*sin(r)=0.5*sin(4*pi)=0` and the noisy-easy line starts at `(0.5/r)*0=0`.

    THE POINT (`docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md` §WP-3): noise is COMMON-MODE across
    widths at a fixed input, so no width fits it down — the noisy-easy per-width MSE curve is FLAT at a
    high level (`~HETERO3_NOISY_SIGMA**2`), and the honest capacity verdict there is "stay NARROW". A
    dial that keys on capacity-hunger reads this correctly; a dial that keys on raw error magnitude
    over-feeds it. This is the width edition of the k-selection program's smooth-negative-control.

    Args:
        n: number of points (scale by 3/2 vs the 2-region toy to keep per-region density equal).
        seed: RNG seed (`np.random.default_rng`).
        r: per-region width; domain is `[-r, 2r)` (default `4*pi`).

    Returns:
        `(x, y, region)`, each shape `(n,)`, `x`/`y` float32; `region` int in `{0, 1, 2}`. Same NON-
        standardized-`x` contract as `make_hetero` (callers standardize on TRAIN stats).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-r, 2.0 * r, n)
    region = np.select([x < 0.0, x < r], [0, 1], default=2)
    y_signal = np.select(
        [region == 0, region == 1],
        [(0.5 / r) * x, 0.5 * np.sin(x)],
        default=(0.5 / r) * (x - r),
    )
    noise_sigma = np.where(x >= r, HETERO3_NOISY_SIGMA, HETERO_NOISE_SIGMA)  # region 2 <=> x >= r
    y = (y_signal + rng.normal(0.0, 1.0, n) * noise_sigma).astype(np.float32)
    return x.astype(np.float32), y, region.astype(int)


def train_nested_width(
    net: NestedWidthNet,
    x_t: torch.Tensor,
    y_t: torch.Tensor,
    n_epochs: int,
    lr: float,
    seed: int,
    device: str = "cpu",
    schedule: WidthSchedule = WidthSchedule.NESTED,
) -> NestedWidthNet:
    """Phase-1 training schedule for the nested-width net: NESTED (W1, default) or SANDWICH (W2).

    `WidthSchedule.NESTED` (unchanged from W1): every epoch draws `k_i ~ Uniform{1..w_max}`
    independently per example, scores each example ONLY at its own drawn width (via `net.forward`'s
    gather), and backprops the mean Gaussian NLL. No selector head is trained here — the draw
    distribution is fixed uniform, mirroring `NestedStrategy`'s "a draw, not a learned selector"
    semantics (layer_selection_strategies.py:187-202) and `capacity_ladder_h1.py::_train_phase1`'s
    training-loop shape (full-batch, one fresh draw per epoch, plain Adam). Trains width=w_max only
    ~1/w_max of the time per epoch — W1's diagnosed under-fit of the hard region at w_max
    (`EXECUTION_PLAN.md` §6: 0.55 nat short of the noise floor vs dedicated nets' ~0.09 nat).

    `WidthSchedule.SANDWICH` (W2 fix, `EXECUTION_PLAN.md` §6): every step, ALWAYS full-batch-scores
    width=1 (min) AND width=w_max (max), PLUS 2 random intermediate widths drawn WITHOUT
    replacement from `{2..w_max-1}` (degrades gracefully to however many candidates exist, e.g. 0
    when `w_max<=2`); sums the 4 (or fewer) per-width mean Gaussian-NLL losses and takes ONE
    optimizer step. Guarantees width=w_max is trained on every single step, not just `1/w_max` of
    them.

    Args:
        net: the `NestedWidthNet` to train, mutated in place.
        x_t: `(N, 1)` training inputs.
        y_t: `(N,)` training targets.
        n_epochs: number of full-batch Adam epochs (== steps; both schedules are full-batch).
        lr: Adam learning rate.
        seed: RNG seed for the per-epoch width draw (reproducibility; does not seed torch's global
            init RNG — caller is responsible for `torch.manual_seed` before constructing `net` if
            deterministic initialization is also required).
        device: torch device string.
        schedule: `WidthSchedule.NESTED` (default — the sinc driver relies on this default and must
            keep using it) or `WidthSchedule.SANDWICH`.

    Returns:
        `net`, moved to `device`, in eval mode, mutated in place (also returned for convenience).
    """
    net.to(device)
    x_t = x_t.to(device)
    y_t = y_t.to(device).reshape(-1)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = x_t.shape[0]
    net.train()

    if schedule is WidthSchedule.NESTED:
        for _ in range(int(n_epochs)):
            opt.zero_grad()
            k_draw = torch.randint(1, net.w_max + 1, (n,), generator=gen).to(device)
            mean, log_var = net(x_t, k_draw)
            ll = gaussian_log_likelihood(mean.squeeze(1), log_var.squeeze(1), y_t)
            loss = -ll.mean()
            loss.backward()
            opt.step()
    elif schedule is WidthSchedule.SANDWICH:
        mid_candidates = list(range(2, net.w_max))  # {2..w_max-1}
        n_mid_draw = min(2, len(mid_candidates))
        for _ in range(int(n_epochs)):
            opt.zero_grad()
            widths = [1, net.w_max]
            if n_mid_draw:
                perm = torch.randperm(len(mid_candidates), generator=gen)[:n_mid_draw]
                widths += [mid_candidates[i] for i in perm.tolist()]
            total_loss = torch.zeros((), device=device)
            for k in widths:
                mean, log_var = net.forward_width(x_t, k)
                ll = gaussian_log_likelihood(mean.squeeze(1), log_var.squeeze(1), y_t)
                total_loss = total_loss + (-ll.mean())
            total_loss.backward()
            opt.step()
    else:
        raise ValueError(f"unknown schedule: {schedule!r}")

    net.eval()
    return net


# ---------------------------------------------------------------------------
# Selftest -- random-init only, no training. MUST pass before any real read.
# ---------------------------------------------------------------------------


def _assert_prefix_invariance(net: NestedWidthNet, x: torch.Tensor) -> tuple[bool, float]:
    """(a) width-k output is invariant to perturbing the readout weights of nodes >= k."""
    ok_all = True
    max_err = 0.0
    for k in range(1, net.w_max + 1):
        with torch.no_grad():
            mean0, logvar0 = net.forward_width(x, k)
        mean_w_orig = net.mean_head.weight.detach().clone()
        logvar_w_orig = net.logvar_head.weight.detach().clone()
        with torch.no_grad():
            if k < net.w_max:
                net.mean_head.weight[:, k:] += torch.randn_like(net.mean_head.weight[:, k:]) * 5.0
                net.logvar_head.weight[:, k:] += torch.randn_like(net.logvar_head.weight[:, k:]) * 5.0
            mean1, logvar1 = net.forward_width(x, k)
            net.mean_head.weight.copy_(mean_w_orig)
            net.logvar_head.weight.copy_(logvar_w_orig)
        err = max((mean0 - mean1).abs().max().item(), (logvar0 - logvar1).abs().max().item())
        max_err = max(max_err, err)
        ok_all = ok_all and (err < _PREFIX_TOL)
    return ok_all, max_err


def _assert_nesting_agreement(net: NestedWidthNet, x: torch.Tensor) -> tuple[bool, float]:
    """(b) width-(k+1) minus width-k equals exactly the newly-included node's own contribution."""
    ok_all = True
    max_err = 0.0
    with torch.no_grad():
        h = net.hidden(x)
    for k in range(1, net.w_max):  # k -> k+1, node index k (0-indexed) newly included
        with torch.no_grad():
            mean_k, logvar_k = net.forward_width(x, k)
            mean_k1, logvar_k1 = net.forward_width(x, k + 1)
            expected_mean_delta = h[:, k] * net.mean_head.weight[0, k]
            expected_logvar_delta = h[:, k] * net.logvar_head.weight[0, k]
        err = max(
            (mean_k1.squeeze(1) - mean_k.squeeze(1) - expected_mean_delta).abs().max().item(),
            (logvar_k1.squeeze(1) - logvar_k.squeeze(1) - expected_logvar_delta).abs().max().item(),
        )
        max_err = max(max_err, err)
        ok_all = ok_all and (err < _NESTING_TOL)
    return ok_all, max_err


def _assert_all_widths_finite(net: NestedWidthNet, x: torch.Tensor) -> tuple[bool, float]:
    """(c) all-widths pass: correct finite shapes, AND agrees with the per-width masked forward."""
    with torch.no_grad():
        mean_all, logvar_all = net.all_widths_forward(x)
    ok_shape = tuple(mean_all.shape) == (x.shape[0], net.w_max) and tuple(logvar_all.shape) == (x.shape[0], net.w_max)
    ok_finite = bool(torch.isfinite(mean_all).all()) and bool(torch.isfinite(logvar_all).all())

    max_err = 0.0
    for k in range(1, net.w_max + 1):
        with torch.no_grad():
            mean_k, logvar_k = net.forward_width(x, k)
        err = max(
            (mean_all[:, k - 1 : k] - mean_k).abs().max().item(),
            (logvar_all[:, k - 1 : k] - logvar_k).abs().max().item(),
        )
        max_err = max(max_err, err)
    ok_consistent = max_err < _CONSISTENCY_TOL

    return (ok_shape and ok_finite and ok_consistent), max_err


def run_selftest() -> bool:
    """Runs all three no-training checks on a randomly-initialized net and prints PASS/FAIL."""
    torch.manual_seed(0)
    net = NestedWidthNet(w_max=6)
    net.eval()
    x = torch.randn(37, 1)

    ok_a, err_a = _assert_prefix_invariance(net, x)
    print(f"[nested_width_net selftest] (a) prefix invariance: max_abs_err={err_a:.3e} (tol={_PREFIX_TOL:.0e})  {'PASS' if ok_a else 'FAIL'}")

    ok_b, err_b = _assert_nesting_agreement(net, x)
    print(f"[nested_width_net selftest] (b) nesting agreement: max_abs_err={err_b:.3e} (tol={_NESTING_TOL:.0e})  {'PASS' if ok_b else 'FAIL'}")

    ok_c, err_c = _assert_all_widths_finite(net, x)
    print(f"[nested_width_net selftest] (c) all-widths pass (shape/finite + masking-vs-cumsum consistency): max_abs_err={err_c:.3e} (tol={_CONSISTENCY_TOL:.0e})")
    print(f"  {'PASS' if ok_c else 'FAIL'}")

    ok = ok_a and ok_b and ok_c
    print(f"[nested_width_net selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Parses args and runs the selftest, or prints help (this module has no standalone real-run mode)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="No-training known-answer checks of the nested-width prefix property.")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
