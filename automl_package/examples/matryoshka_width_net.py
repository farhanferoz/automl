"""Matryoshka width net — the fallback arm: shared trunk, per-rung DEDICATED readout heads.

(`docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md` §2.7, §2.8, §4.2)

Shared trunk `Linear(1 -> w_max) -> tanh` gives `h(x) in R^{w_max}`; for EACH rung `k = 1..w_max` a
DEDICATED pair of heads `mean_head_k, logvar_head_k = Linear(k -> 1)` reads off the prefix
`h[:, :k]`. This removes `nested_width_net.NestedWidthNet`'s shared-readout entanglement -- the
mechanism Matryoshka Representation Learning's (Kusupati et al. 2022) no-degradation result rides
on -- but it does NOT give the shared trunk's hidden units stable identity or ordering: every rung's
gradient hits the trunk jointly, forever. So the coherence-invariant scorecard here is: (2)
self-contained partially fixed (each rung has its own readout basis), (1)/(3) stable-identity /
importance-ordering NOT fixed (plan §2.8 table) -- which is exactly why this is the FALLBACK arm and
the comparison to `cascade_width_net.ResidualCascadeNet` is informative: it isolates whether frozen
identity + guaranteed ordering matter beyond readout disentanglement alone.

`train_matryoshka` (module-level, below) trains ALL rungs JOINTLY with ONE optimizer: every step,
the loss is the UNWEIGHTED SUM of all `w_max` rungs' mean Gaussian NLL (MRL's per-rung relative-
importance weights `c_m` are an unexplored dial here -- plan §2.7 pre-registers uniform `c_k = 1`).
Convergence is gated PER RUNG (`convergence.ConvergenceTracker`, one per rung, mirroring
`kdropout_converged_width_experiment._train_kdropout_to_convergence`'s joint-training loop shape);
the loop stops once every rung's tracker has flattened or the safety cap is hit. Because the trunk is
SHARED across rungs (unlike `IndependentWidthNet`'s disjoint per-width sub-nets), per-rung weight
restoration is ill-defined -- there is only ONE net. Instead a single extra tracker watches the
SUMMED val NLL across all rungs and snapshots the WHOLE net's `state_dict` on improvement; that
best-joint snapshot is restored at the end (`"best_restore": "joint_sum"`, recorded by the driver so
the comparability caveat vs the disjoint W batteries stays explicit).

Selftest (`--selftest`, no training, random init only):
  (a) rung-k output is invariant to perturbing trunk hidden units `>= k` AND any OTHER rung's heads
      (`!= k`) -- the two independence properties a per-rung-head architecture must have.
  (b) `all_widths_forward` agrees with the per-k `forward_width` loop (tol 1e-5).
  (c) `all_widths_forward` returns finite `(mean, log_var)` of shape `(N, w_max)` for both heads.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/matryoshka_width_net.py --selftest
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import convergence as cvg  # noqa: E402
import nested_width_net as nwn  # noqa: E402

W_MAX_DEFAULT = 12
LR_DEFAULT = 1e-2  # plan §4.0: LR 1e-2 Adam everywhere.

_PREFIX_PERTURB_TOL = 1e-6
_CONSISTENCY_TOL = 1e-5
_PERTURB_SCALE = 5.0


class MatryoshkaWidthNet(nn.Module):
    """Shared `Linear(1 -> w_max)` trunk, tanh, with `w_max` DEDICATED `(mean, log_var)` head pairs.

    Rung k's heads are `Linear(k -> 1)`, reading only the first `k` trunk hidden units -- so every
    rung is still a valid prefix (cheap prefix inference), but unlike `NestedWidthNet` each rung owns
    its OWN readout parameters instead of sharing one `Linear(w_max -> 1)`.
    """

    def __init__(self, w_max: int = W_MAX_DEFAULT, activation: type[nn.Module] = nn.Tanh) -> None:
        """Builds the shared trunk plus `w_max` dedicated `(mean_head_k, logvar_head_k)` pairs.

        Args:
            w_max: maximum hidden width / largest rung.
            activation: hidden-layer nonlinearity class (instantiated with no args); tanh per the
                plan's fixed hyperparameter (§4.0).
        """
        super().__init__()
        self.w_max = int(w_max)
        self.trunk = nn.Linear(1, self.w_max)
        self.act = activation()
        self.mean_heads = nn.ModuleList(nn.Linear(k, 1) for k in range(1, self.w_max + 1))
        self.logvar_heads = nn.ModuleList(nn.Linear(k, 1) for k in range(1, self.w_max + 1))

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, 1) -> (N, w_max)` post-activation hidden representation (the trunk's full output)."""
        return self.act(self.trunk(x))

    def forward_width(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at rung `k` (1..w_max) via rung k's OWN dedicated head pair, each `(N, 1)`."""
        if not (1 <= k <= self.w_max):
            raise ValueError(f"k={k} out of range [1, {self.w_max}]")
        h = self.hidden(x)[:, :k]
        return self.mean_heads[k - 1](h), self.logvar_heads[k - 1](h)

    def all_widths_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at every rung `k=1..w_max`, each shape `(N, w_max)`.

        No cumsum trick (unlike `NestedWidthNet`/`ResidualCascadeNet`) -- each rung's heads are
        independent parameters, not a linear function of a shared readout, so this is `w_max` small
        `(k -> 1)` matmuls, one per rung.
        """
        means, logvars = [], []
        for k in range(1, self.w_max + 1):
            mean_k, logvar_k = self.forward_width(x, k)
            means.append(mean_k)
            logvars.append(logvar_k)
        return torch.cat(means, dim=1), torch.cat(logvars, dim=1)


def train_matryoshka(
    net: MatryoshkaWidthNet,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    max_epochs: int,
    check_every: int,
    patience: int,
    min_delta: float,
    lr: float = LR_DEFAULT,
) -> dict[int, cvg.ConvergenceResult]:
    """Joint full-batch training: one Adam optimizer, loss = unweighted sum of every rung's mean NLL.

    Per-rung convergence is gated by its OWN `ConvergenceTracker` (verdicts/trajectories only); an
    extra tracker on the SUMMED val NLL across all rungs snapshots the whole net's `state_dict` on
    improvement and restores that best-joint snapshot at the end (§4.2 -- per-rung restoration is
    ill-defined because the trunk is shared, unlike `IndependentWidthNet`'s disjoint sub-nets).

    Args:
        net: the `MatryoshkaWidthNet` to train in place.
        x_tr: standardized training inputs.
        y_tr: standardized training targets.
        x_val: standardized held-out inputs, used only for convergence monitoring.
        y_val: standardized held-out targets, used only for convergence monitoring.
        max_epochs: safety cap on optimizer steps (== epochs; full-batch, joint over all rungs).
        check_every: epochs between per-rung held-out checkpoints, passed to each rung's tracker.
        patience: flat checkpoints that declare a rung converged, passed to each rung's tracker.
        min_delta: held-out-loss decrease (nats) counted as improvement, passed to each rung's
            `cvg.ConvergenceTracker` (including the extra joint-sum tracker).
        lr: Adam learning rate (plan §4.0 default 1e-2).

    Returns:
        `{rung k -> cvg.ConvergenceResult}` for k = 1..w_max, best-JOINT weights already restored.
    """
    x_tr = x_tr.reshape(-1, 1)
    x_val = x_val.reshape(-1, 1)
    y_tr = y_tr.reshape(-1)
    y_val = y_val.reshape(-1)
    w_max = net.w_max

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    trackers = {k: cvg.ConvergenceTracker(patience=patience, min_delta=min_delta) for k in range(1, w_max + 1)}
    joint_tracker = cvg.ConvergenceTracker(patience=patience, min_delta=min_delta)
    best_joint_state: dict | None = None

    net.train()
    final_epoch = max_epochs
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        total_loss = torch.zeros(())
        for k in range(1, w_max + 1):
            mean, log_var = net.forward_width(x_tr, k)
            ll = nwn.gaussian_log_likelihood(mean.squeeze(1), log_var.squeeze(1), y_tr)
            total_loss = total_loss + (-ll.mean())
        total_loss.backward()
        opt.step()

        if epoch % check_every == 0:
            net.eval()
            with torch.no_grad():
                val_sum = 0.0
                for k in range(1, w_max + 1):
                    mean, log_var = net.forward_width(x_val, k)
                    v = float((-nwn.gaussian_log_likelihood(mean.squeeze(1), log_var.squeeze(1), y_val)).mean().item())
                    trackers[k].update(epoch, v)
                    val_sum += v
                if joint_tracker.update(epoch, val_sum):
                    best_joint_state = {n: t.detach().clone() for n, t in net.state_dict().items()}
            net.train()
            if all(t.done for t in trackers.values()):
                final_epoch = epoch
                break

    if best_joint_state is not None:
        net.load_state_dict(best_joint_state)
    net.eval()
    return {k: trackers[k].result(final_epoch=final_epoch) for k in range(1, w_max + 1)}


# ---------------------------------------------------------------------------
# Selftest -- random-init only, no training. MUST pass before any real read.
# ---------------------------------------------------------------------------


def _assert_prefix_invariance(net: MatryoshkaWidthNet, x: torch.Tensor) -> tuple[bool, float]:
    """(a) rung-k output invariant to perturbing trunk hidden units >= k AND any other rung's heads."""
    ok_all = True
    max_err = 0.0
    for k in range(1, net.w_max + 1):
        with torch.no_grad():
            mean0, logvar0 = net.forward_width(x, k)

        trunk_w_orig = net.trunk.weight.detach().clone()
        trunk_b_orig = net.trunk.bias.detach().clone()
        other = [j for j in range(net.w_max) if j != k - 1]
        head_snapshots = {
            j: (
                net.mean_heads[j].weight.detach().clone(),
                net.mean_heads[j].bias.detach().clone(),
                net.logvar_heads[j].weight.detach().clone(),
                net.logvar_heads[j].bias.detach().clone(),
            )
            for j in other
        }

        with torch.no_grad():
            if k < net.w_max:
                net.trunk.weight[k:, :] += torch.randn_like(net.trunk.weight[k:, :]) * _PERTURB_SCALE
                net.trunk.bias[k:] += torch.randn_like(net.trunk.bias[k:]) * _PERTURB_SCALE
            for j in other:
                net.mean_heads[j].weight += torch.randn_like(net.mean_heads[j].weight) * _PERTURB_SCALE
                net.mean_heads[j].bias += torch.randn_like(net.mean_heads[j].bias) * _PERTURB_SCALE
                net.logvar_heads[j].weight += torch.randn_like(net.logvar_heads[j].weight) * _PERTURB_SCALE
                net.logvar_heads[j].bias += torch.randn_like(net.logvar_heads[j].bias) * _PERTURB_SCALE
            mean1, logvar1 = net.forward_width(x, k)
            net.trunk.weight.copy_(trunk_w_orig)
            net.trunk.bias.copy_(trunk_b_orig)
            for j, (mw, mb, lw, lb) in head_snapshots.items():
                net.mean_heads[j].weight.copy_(mw)
                net.mean_heads[j].bias.copy_(mb)
                net.logvar_heads[j].weight.copy_(lw)
                net.logvar_heads[j].bias.copy_(lb)

        err = max((mean0 - mean1).abs().max().item(), (logvar0 - logvar1).abs().max().item())
        max_err = max(max_err, err)
        ok_all = ok_all and (err < _PREFIX_PERTURB_TOL)
    return ok_all, max_err


def _assert_all_widths_consistency(net: MatryoshkaWidthNet, x: torch.Tensor) -> tuple[bool, float]:
    """(b) all_widths_forward agrees with the per-k forward_width loop (tol 1e-5)."""
    with torch.no_grad():
        mean_all, logvar_all = net.all_widths_forward(x)
    max_err = 0.0
    for k in range(1, net.w_max + 1):
        with torch.no_grad():
            mean_k, logvar_k = net.forward_width(x, k)
        err = max((mean_all[:, k - 1 : k] - mean_k).abs().max().item(), (logvar_all[:, k - 1 : k] - logvar_k).abs().max().item())
        max_err = max(max_err, err)
    return max_err < _CONSISTENCY_TOL, max_err


def _assert_finite_shapes(net: MatryoshkaWidthNet, x: torch.Tensor) -> bool:
    """(c) all_widths_forward returns finite (N, w_max) for both heads."""
    with torch.no_grad():
        mean_all, logvar_all = net.all_widths_forward(x)
    ok_shape = tuple(mean_all.shape) == (x.shape[0], net.w_max) and tuple(logvar_all.shape) == (x.shape[0], net.w_max)
    ok_finite = bool(torch.isfinite(mean_all).all()) and bool(torch.isfinite(logvar_all).all())
    return ok_shape and ok_finite


def run_selftest() -> bool:
    """Runs all three no-training checks on a randomly-initialized net and prints PASS/FAIL."""
    torch.manual_seed(0)
    net = MatryoshkaWidthNet(w_max=6)
    net.eval()
    x = torch.randn(37, 1)

    ok_a, err_a = _assert_prefix_invariance(net, x)
    print(f"[matryoshka selftest] (a) prefix invariance (trunk>=k, heads!=k): max_abs_err={err_a:.3e} (tol={_PREFIX_PERTURB_TOL:.0e})  {'PASS' if ok_a else 'FAIL'}")

    ok_b, err_b = _assert_all_widths_consistency(net, x)
    print(f"[matryoshka selftest] (b) all-widths vs per-k consistency: max_abs_err={err_b:.3e} (tol={_CONSISTENCY_TOL:.0e})  {'PASS' if ok_b else 'FAIL'}")

    ok_c = _assert_finite_shapes(net, x)
    print(f"[matryoshka selftest] (c) all-widths shape/finite: {'PASS' if ok_c else 'FAIL'}")

    ok = ok_a and ok_b and ok_c
    print(f"[matryoshka selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Parses args and runs the selftest, or prints help (this module has no standalone real-run mode)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="No-training known-answer checks of the per-rung-head architecture.")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
