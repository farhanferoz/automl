"""Convergence-gated training: decide epochs from the loss TRAJECTORY, never a fixed count.

Standing rule (agent-memory `feedback_check_loss_trajectory_before_concluding`): every training run
logs its held-out loss trajectory and carries a `converged` flag decided by patience-based early
stopping from that trajectory — NOT by picking an epoch count and reading the endpoint. No conclusion
may be drawn from a run whose `converged` flag is False (the only valid read is "needs more training").

`fit_to_convergence` runs a caller-supplied training step until the held-out loss stops improving
(patience) or a safety cap is hit, keeping the best-so-far weights and returning the full trajectory.
It is deliberately model-agnostic (the caller closes over the net/optimizer/data and passes a
`step_fn` + `val_loss_fn`) so the SAME gate applies to every model we train (per-width, per-depth, …).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

# Defaults tuned to the width-net dynamics observed in the epoch sweep (very slow late creep on the
# hard middle widths): check often, be patient, count only non-trivial held-out-loss decreases.
DEFAULT_CHECK_EVERY = 500  # epochs between held-out-loss checkpoints
DEFAULT_PATIENCE = 12  # checkpoints with no >min_delta improvement before declaring convergence
DEFAULT_MIN_DELTA = 2e-3  # held-out-loss decrease (nats) that counts as a real improvement
DEFAULT_MAX_EPOCHS = 60000  # safety cap; hitting it with the loss still falling => NOT converged
_STILL_IMPROVING_EPS = 5e-3  # residual improvement over the patience window above which we warn "slow creep"
_MIN_TRAJ_POINTS_FOR_SLOPE = 2  # fewer checkpoints than this => can't estimate a recent-improvement slope

DIVERGENCE_ABS_EPS = 0.2  # nats; mandatory absolute floor (a pure ratio misfires near zero best_val)
DIVERGENCE_REL_FACTOR = 0.5  # divergence threshold = max(DIVERGENCE_ABS_EPS, this * best_val)


@dataclass
class ConvergenceResult:
    """Outcome of one convergence-gated training run — carries the evidence, not just an endpoint."""

    trajectory: list[tuple[int, float]] = field(default_factory=list)  # (epoch, held-out loss) checkpoints
    converged: bool = False  # True = patience-stopped (held-out loss flattened)
    hit_cap: bool = False  # True = stopped only because max_epochs was reached (loss may still be falling)
    stop_epoch: int = 0
    best_val: float = math.inf
    best_epoch: int = 0
    recent_improvement: float = 0.0  # held-out-loss decrease over the last `patience` checkpoints

    @property
    def still_improving(self) -> bool:
        """True if the held-out loss was still falling appreciably over the final window (a slow-creep stop)."""
        return self.recent_improvement > _STILL_IMPROVING_EPS

    @property
    def diverged(self) -> bool:
        """True if the loss exploded past its best and never recovered back within tolerance.

        Patience-based stopping alone can certify a run `trustworthy` even when its held-out loss spiked
        then plateaued (or is still declining from) a much worse level than its best epoch — best-weights
        restore limits the reported-METRIC damage, but the flag itself would still lie. Compares the last
        trajectory point to `best_val` against a floor that is the larger of an absolute epsilon and a
        fraction of `best_val`; the absolute floor is mandatory because a pure ratio misfires near zero
        (e.g. best 0.0015 with a harmless 0.003 final => ratio 3, a false positive without the floor).
        """
        if not self.trajectory:
            return False
        final_trajectory_point_loss = self.trajectory[-1][1]
        return (final_trajectory_point_loss - self.best_val) > max(DIVERGENCE_ABS_EPS, DIVERGENCE_REL_FACTOR * self.best_val)

    @property
    def trustworthy(self) -> bool:
        """A result is safe to CONCLUDE from only if it flattened, was not still visibly creeping, and did not diverge."""
        return self.converged and not self.hit_cap and not self.still_improving and not self.diverged

    def summary(self) -> dict:
        """JSON-able summary (trajectory kept in full so the curve is always inspectable)."""
        return {
            "converged": self.converged,
            "hit_cap": self.hit_cap,
            "trustworthy": self.trustworthy,
            "still_improving_at_stop": self.still_improving,
            "diverged": self.diverged,
            "stop_epoch": self.stop_epoch,
            "best_val": self.best_val,
            "best_epoch": self.best_epoch,
            "recent_improvement": self.recent_improvement,
            "trajectory": [[int(e), float(v)] for e, v in self.trajectory],
        }


class ConvergenceTracker:
    """Running best/patience state machine for ONE held-out-loss series.

    Extracted from `fit_to_convergence` so single-series training (that function) and JOINT
    multi-series training share IDENTICAL flag semantics. In joint training (e.g. k-dropout, where one
    optimizer step updates several widths at once) each width needs its own convergence verdict from its
    own held-out trajectory; giving each width a tracker guarantees its `trustworthy` flag means exactly
    the same thing it would if that width had been trained alone.
    """

    def __init__(self, *, patience: int = DEFAULT_PATIENCE, min_delta: float = DEFAULT_MIN_DELTA) -> None:
        """Args: `patience`/`min_delta` as in `fit_to_convergence` (same defaults, same meaning)."""
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_val = math.inf
        self.best_epoch = 0
        self.since_improve = 0
        self.trajectory: list[tuple[int, float]] = []
        self.converged = False
        self.stop_epoch = 0

    def update(self, epoch: int, val: float) -> bool:
        """Record one held-out checkpoint; return True iff it is a NEW best (caller snapshots weights then)."""
        val = float(val)
        self.trajectory.append((int(epoch), val))
        improved = val < self.best_val - self.min_delta
        if improved:
            self.best_val = val
            self.best_epoch = int(epoch)
            self.since_improve = 0
        else:
            self.since_improve += 1
        if not self.converged and self.since_improve >= self.patience:
            self.converged = True
            self.stop_epoch = int(epoch)
        return improved

    @property
    def done(self) -> bool:
        """True once the series has flattened (patience reached) — the caller may stop its loop."""
        return self.converged

    def result(self, *, final_epoch: int) -> ConvergenceResult:
        """Freeze into a `ConvergenceResult`.

        `final_epoch` is the loop's last epoch, used as the stop epoch when the series never converged
        (i.e. hit the cap).
        """
        recent_improvement = 0.0
        if len(self.trajectory) > self.patience:
            recent_improvement = float(self.trajectory[-(self.patience + 1)][1] - self.trajectory[-1][1])
        elif len(self.trajectory) >= _MIN_TRAJ_POINTS_FOR_SLOPE:
            recent_improvement = float(self.trajectory[0][1] - self.trajectory[-1][1])
        return ConvergenceResult(
            trajectory=self.trajectory,
            converged=self.converged,
            hit_cap=not self.converged,
            stop_epoch=self.stop_epoch if self.converged else int(final_epoch),
            best_val=self.best_val,
            best_epoch=self.best_epoch,
            recent_improvement=recent_improvement,
        )


def fit_to_convergence(
    train_module: nn.Module,
    step_fn: Callable[[], None],
    val_loss_fn: Callable[[], float],
    *,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    check_every: int = DEFAULT_CHECK_EVERY,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
) -> ConvergenceResult:
    """Train until the held-out loss flattens (patience) or `max_epochs` is hit; keep the best weights.

    Args:
        train_module: the module whose parameters `step_fn` optimizes; its `state_dict` is snapshotted at
            each held-out-loss improvement and restored at the end (best-weights, not last-weights).
        step_fn: performs exactly ONE training step (zero_grad → loss → backward → optimizer step). The
            caller owns the net/optimizer/data closure.
        val_loss_fn: returns the current HELD-OUT loss (lower = better); called every `check_every` epochs.
        max_epochs: safety cap. Reaching it means the loss never flattened → `hit_cap=True`, `converged=False`.
        check_every: epochs between held-out-loss checkpoints (the trajectory resolution).
        patience: number of consecutive checkpoints without a >`min_delta` improvement that declares convergence.
        min_delta: minimum held-out-loss decrease (nats) counted as a real improvement (noise guard).

    Returns:
        A `ConvergenceResult` with the full trajectory, the `converged`/`hit_cap` flags, and the best weights
        already loaded back into `train_module`.
    """
    tracker = ConvergenceTracker(patience=patience, min_delta=min_delta)
    best_state: dict | None = None
    final_epoch = max_epochs

    train_module.train()
    for epoch in range(1, max_epochs + 1):
        step_fn()
        if epoch % check_every == 0:
            if tracker.update(epoch, val_loss_fn()):
                best_state = {k: t.detach().clone() for k, t in train_module.state_dict().items()}
            if tracker.done:
                final_epoch = epoch
                break

    if best_state is not None:
        train_module.load_state_dict(best_state)
    return tracker.result(final_epoch=final_epoch)


def format_trajectory(result: ConvergenceResult, label: str = "") -> str:
    """One-line-per-run readable trajectory + verdict, so the curve is SHOWN, never just summarized."""
    pts = " ".join(f"{e}:{v:.3f}" for e, v in result.trajectory)
    verdict = "CONVERGED" if result.trustworthy else ("HIT_CAP(needs more)" if result.hit_cap else "slow-creep(not trustworthy)")
    return f"{label} [{verdict}] best={result.best_val:.4f}@{result.best_epoch} stop@{result.stop_epoch} recent_impr={result.recent_improvement:.4f}\n    traj: {pts}"
