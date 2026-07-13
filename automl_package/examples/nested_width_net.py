"""W1 — nested-width probabilistic net: the WIDTH analog of the depth lane's NESTED schedule.

(docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md §2a)

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
import torch.nn as nn

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

W_MAX_DEFAULT = 16
_PREFIX_TOL = 1e-6
_NESTING_TOL = 1e-5
_CONSISTENCY_TOL = 1e-5

HETERO_NOISE_SIGMA = 0.05
HETERO_R_DEFAULT = 4.0 * math.pi


class WidthSchedule(enum.Enum):
    """Which per-step width-sampling schedule `train_nested_width` uses (closed set, not a string).

    `docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md` §6.
    """

    NESTED = "nested"  # W1: per-example uniform width draw each epoch (default; sinc keeps using this).
    SANDWICH = "sandwich"  # W2 fix: every step ALWAYS trains width=1 and width=w_max (+2 random mid).


class NestedWidthNet(nn.Module):
    """Single-hidden-layer probabilistic net with a nested-width readout.

    `Linear(1 -> w_max)` -> activation -> hidden vector `h`; `mean_head`/`logvar_head` are
    `Linear(w_max -> 1)`. A width-k forward zeros `h[:, k:]` before both readouts.
    """

    def __init__(self, w_max: int = W_MAX_DEFAULT, activation: type[nn.Module] = nn.Tanh) -> None:
        """Builds the trunk + the two (mean, log-variance) readout heads.

        Args:
            w_max: maximum hidden width (the largest prefix the net can express).
            activation: hidden-layer nonlinearity class (instantiated with no args); tanh per
                the plan's fixed hyperparameter (`docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md`
                §2d config).
        """
        super().__init__()
        self.w_max = int(w_max)
        self.trunk = nn.Linear(1, self.w_max)
        self.activation = activation()
        self.mean_head = nn.Linear(self.w_max, 1)
        self.logvar_head = nn.Linear(self.w_max, 1)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, 1) -> (N, w_max)` post-activation hidden representation (the trunk's full output)."""
        return self.activation(self.trunk(x))

    def forward_width(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at one FIXED width `k` (1..w_max), each shape `(N, 1)`.

        Explicit masking (`h[:, k:] = 0` before both readouts) — the literal statement of the
        width-k prefix property; `all_widths_forward` computes the same values for every k at
        once via an algebraically equivalent cumulative sum (selftest (c) checks the two agree).
        """
        if not (1 <= k <= self.w_max):
            raise ValueError(f"k={k} out of range [1, {self.w_max}]")
        h = self.hidden(x)
        mask = torch.zeros_like(h)
        mask[:, :k] = 1.0
        h_masked = h * mask
        return self.mean_head(h_masked), self.logvar_head(h_masked)

    def all_widths_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at EVERY width `k=1..w_max` in one pass; each shape `(N, w_max)`.

        Column `k-1` is width-k's readout. Both readouts are linear in `h`, so zeroing columns
        `k..` before a `Linear(w_max -> 1)` is exactly `sum_{i<k} h[:, i] * weight[0, i] + bias`
        — a per-node contribution followed by a cumulative sum over the first k nodes. Exact and
        vectorized, no python loop over widths (the width analog of `NestedStrategy.
        all_depth_outputs`'s "one forward pass, every depth" cached-forward trick).
        """
        h = self.hidden(x)  # (N, w_max)
        mean_contrib = h * self.mean_head.weight.squeeze(0)  # (N, w_max), elementwise per-node contribution
        logvar_contrib = h * self.logvar_head.weight.squeeze(0)
        mean_all = torch.cumsum(mean_contrib, dim=1) + self.mean_head.bias
        logvar_all = torch.cumsum(logvar_contrib, dim=1) + self.logvar_head.bias
        return mean_all, logvar_all

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample width draw: `k` is an `(N,)` int tensor in `[1, w_max]`.

        Gathers each sample's own drawn-width readout out of `all_widths_forward` — the free
        reshape of one forward pass that `NestedStrategy.forward` relies on for depth
        (layer_selection_strategies.py:265-269), here over width columns instead of depth blocks.

        Returns:
            `(mean, log_var)`, each shape `(N, 1)`.
        """
        mean_all, logvar_all = self.all_widths_forward(x)
        col = (k - 1).unsqueeze(1)
        mean = mean_all.gather(1, col)
        logvar = logvar_all.gather(1, col)
        return mean, logvar


class IndependentWidthNet(nn.Module):
    """W4: K DISJOINT one-hidden-layer probabilistic sub-nets, one per width — NO weight sharing.

    The width twin of `automl_package/models/independent_weights_flexible_neural_network.py`
    (`IndependentWeightsFlexibleNN`: independent weights per DEPTH, one `nn.ModuleList` entry per
    depth, trained with the same k-dropout selection). Here width-k is its OWN sub-net
    `Linear(1 -> k) -> tanh -> (mean_head_k, logvar_head_k)`, sharing NOTHING with width-k'≠k. This
    deliberately breaks the shared-trunk prefix property of `NestedWidthNet` — the one variable W1/W2
    pinned as the obstruction (`docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md` §6 follow-up):
    node-0 no longer has to double as a good small-width predictor, so each width is free to fit to
    its own ceiling.

    Exposes the SAME `w_max` / `forward_width` / `all_widths_forward` interface as `NestedWidthNet`,
    so it is a drop-in for `sinc_width_experiment._score_all_widths` and for the SANDWICH branch of
    `train_nested_width` (which touches only `net.w_max` and `net.forward_width`) — the ONLY thing
    that changes vs the shared W2 run is shared-vs-independent weights.
    """

    def __init__(self, w_max: int = W_MAX_DEFAULT, activation: type[nn.Module] = nn.Tanh) -> None:
        """Builds `w_max` disjoint sub-nets; sub-net k (0-indexed k-1) has exactly k hidden units."""
        super().__init__()
        self.w_max = int(w_max)
        self.subnets = nn.ModuleList(
            nn.ModuleDict({"trunk": nn.Linear(1, k), "act": activation(), "mean_head": nn.Linear(k, 1), "logvar_head": nn.Linear(k, 1)})
            for k in range(1, self.w_max + 1)
        )

    def forward_width(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` from width-k's OWN sub-net (uses no other width's weights). Each `(N, 1)`."""
        if not (1 <= k <= self.w_max):
            raise ValueError(f"k={k} out of range [1, {self.w_max}]")
        sub = self.subnets[k - 1]
        h = sub["act"](sub["trunk"](x))
        return sub["mean_head"](h), sub["logvar_head"](h)

    def all_widths_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at every width, each `(N, w_max)`; column k-1 is width-k's own sub-net.

        No cumsum trick here (weights are NOT nested/shared) — just each independent sub-net's output
        concatenated. Matches `NestedWidthNet.all_widths_forward`'s output shape/semantics so the
        shared scoring/selector code path is identical.
        """
        means, logvars = [], []
        for k in range(1, self.w_max + 1):
            mean_k, logvar_k = self.forward_width(x, k)
            means.append(mean_k)
            logvars.append(logvar_k)
        return torch.cat(means, dim=1), torch.cat(logvars, dim=1)


def gaussian_log_likelihood(mean: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-example Gaussian log-likelihood — same formula as `capacity_ladder_f2._fixed_depth_log_likelihood`."""
    variance = torch.exp(log_var)
    return -0.5 * (math.log(2.0 * math.pi) + log_var + (y - mean) ** 2 / variance)


def make_hetero(n: int, seed: int, r: float = HETERO_R_DEFAULT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """W2 heterogeneous toy: a flat-easy line spliced to a width-hungry-but-learnable sine (§6 W2 toy).

    `x ~ Uniform(-r, r)`. Easy region (`x < 0`): `y = (0.5/r) * x` (a straight line — width-flat,
    1 node suffices). Hard region (`x >= 0`): `y = 0.5 * sin(x)` (2 native-frequency periods over
    `[0, r]` at the default `r=4*pi`; spreading the periods over a wide range keeps the input-space
    frequency ~1, which a small net can learn — unlike a packed `sin(2*pi*P*x)` on `[0,1]`, which
    is UNLEARNABLE by a small net, the same spectral-bias wall as the tent map, confirmed in
    `scratchpad/hetero_toy_probe.py` for P=2,3,4). Both branches are 0 at `x=0`, so `y`'s noise-free
    signal is continuous there. Gaussian noise, `sigma=HETERO_NOISE_SIGMA`, is added to `y`. Probed
    learnable with a clean per-region width gradient in `scratchpad/hetero_toy_probe_v2.py` (easy
    flat ~1.2-2x the noise floor at every width; hard 52x(w=1) -> 1.3x(w=10)).

    Args:
        n: number of points.
        seed: RNG seed (`np.random.default_rng`).
        r: domain half-width; `x ~ Uniform(-r, r)` (default `4*pi`, per the plan's W2 toy spec).

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
    y = (y_signal + rng.normal(0.0, HETERO_NOISE_SIGMA, n)).astype(np.float32)
    region = (x >= 0).astype(int)
    return x.astype(np.float32), y, region


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
