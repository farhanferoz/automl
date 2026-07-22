"""Thin re-export shim -- `MatryoshkaWidthNet` IS the certified `SharedTrunkPerWidthHeadNet`.

WSEL-17 Step 1 (`docs/plans/capacity_programme/width.md` ~2292-2301, §3.9 finding 1) measured the
§3.9 duplicate pair by construction, not by taste: this file's per-rung dedicated head
(`Linear(k -> 1)` reading the raw prefix `h[:, :k]`) and `SharedTrunkPerWidthHeadNet`'s per-width
head (`Linear(w_max -> 1)` reading a MASKED hidden vector, columns `>= k` zeroed) compute the SAME
function -- the certified head's `w_max - k` extra weight columns can never influence a width-`k`
output, because the input they would multiply is already zero. **They differ ONLY in nominal
parameter count**, which `params_effective` (§3.9) already accounts for. Proof, including the
prove-it-fails direction: `tests/test_matryoshka_equivalence.py`.

Consolidation lands HERE: this module no longer defines its own architecture.
`MatryoshkaWidthNet` is now the certified class itself, re-exported under its historical name so
`cascade_width_experiment.py`'s `mwn.MatryoshkaWidthNet(w_max=w_max)` /
`mwn.train_matryoshka(net, ...)` calls keep resolving unchanged -- the `automl_package/examples/
convergence.py` precedent: move the logic, leave the shim, do not rewrite callers. (A shim over a
`PROTECTED.tsv` path is explicitly sanctioned there -- rule 2: "a shim is not a deletion".)

`train_matryoshka` (module-level, below) is UNCHANGED example-script protocol, not duplicated
architecture: the joint all-rungs training loop (one optimizer, unweighted summed loss over every
rung, per-rung `ConvergenceTracker`) is not the thing Step 1 found duplicated -- only the class was.
It stays on the examples side for the same reason `automl_package/models/flexnn/width/
architectures.py`'s module docstring keeps `train_nested_width` there: "experiment protocol, not
library architecture."

**Variance status (§3.7 / WSEL-17 Step 3):** inherited from `SharedTrunkPerWidthHeadNet` -- `log_var`
is a dummy zero, never in the loss graph, so this class provably CANNOT fit a variance.
`train_matryoshka`'s Gaussian-NLL loss therefore reduces to a fixed-unit-variance objective (a
constant-scale reparametrisation of MSE), never a learned one -- it satisfies §3.7 trivially, the
same way the certified class's own docstring already records.

The old per-rung selftest (`_assert_prefix_invariance` etc.) tested THIS module's now-deleted
duplicate architecture (it referenced `logvar_heads`, a per-rung head pair `SharedTrunkPerWidthHeadNet`
does not have -- MSE-only, one dummy-zero `log_var`, not a second head). That coverage is superseded
by `tests/test_matryoshka_equivalence.py` (the equivalence proof) and the certified class's own
existing tests (`tests/test_fused_heads_equivalence.py`, `tests/test_flexible_width_network.py`).
`--selftest` below is now a minimal alias/smoke check, not an architecture proof.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/matryoshka_width_net.py --selftest
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import convergence as cvg  # noqa: E402
import nested_width_net as nwn  # noqa: E402

from automl_package.models.architectures.nested_width_net import SharedTrunkPerWidthHeadNet  # noqa: E402

MatryoshkaWidthNet = SharedTrunkPerWidthHeadNet

__all__ = ["MatryoshkaWidthNet", "train_matryoshka"]

W_MAX_DEFAULT = 12
LR_DEFAULT = 1e-2  # plan §4.0: LR 1e-2 Adam everywhere.


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
        net: the `MatryoshkaWidthNet` (the certified `SharedTrunkPerWidthHeadNet`, re-exported under
            this historical name) to train in place.
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
# Selftest -- minimal alias/smoke check now that the architecture proof lives in
# tests/test_matryoshka_equivalence.py. Random init only, no training.
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Confirms the re-export alias resolves and produces finite output.

    The equivalence proof itself lives in `tests/test_matryoshka_equivalence.py` (including its
    prove-it-fails direction).
    """
    ok = MatryoshkaWidthNet is SharedTrunkPerWidthHeadNet
    print(f"[matryoshka selftest] (a) MatryoshkaWidthNet is the certified SharedTrunkPerWidthHeadNet: {'PASS' if ok else 'FAIL'}")

    torch.manual_seed(0)
    net = MatryoshkaWidthNet(w_max=6)
    net.eval()
    x = torch.randn(37, 1)
    with torch.no_grad():
        mean_all, logvar_all = net.all_widths_forward(x)
    ok_finite = bool(torch.isfinite(mean_all).all()) and bool(torch.isfinite(logvar_all).all())
    print(f"[matryoshka selftest] (b) all_widths_forward finite: {'PASS' if ok_finite else 'FAIL'}")

    ok = ok and ok_finite
    print(f"[matryoshka selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Parses args and runs the selftest, or prints help (this module has no standalone real-run mode)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Minimal alias/smoke check -- see tests/test_matryoshka_equivalence.py for the architecture proof.")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
