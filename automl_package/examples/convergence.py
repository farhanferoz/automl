"""Convergence-gated training: decide epochs from the loss TRAJECTORY, never a fixed count.

Standing rule (agent-memory `feedback_check_loss_trajectory_before_concluding`): every training run
logs its held-out loss trajectory and carries a `converged` flag decided by patience-based early
stopping from that trajectory — NOT by picking an epoch count and reading the endpoint. No conclusion
may be drawn from a run whose `converged` flag is False (the only valid read is "needs more training").

`fit_to_convergence` runs a caller-supplied training step until the held-out loss stops improving
(patience) or a safety cap is hit, keeping the best-so-far weights and returning the full trajectory.
It is deliberately model-agnostic (the caller closes over the net/optimizer/data and passes a
`step_fn` + `val_loss_fn`) so the SAME gate applies to every model we train (per-width, per-depth, …).

Selftest:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/convergence.py --selftest
"""

from __future__ import annotations

import argparse
import math
import sys
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
_TEST_TOL = 1e-6  # selftest float-compare tolerance

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


# ---------------------------------------------------------------------------
# Selftest — scripted held-out-loss sequences (no real training), known answers.
# ---------------------------------------------------------------------------


def _scripted_run(vals: list[float], *, check_every: int = 1, patience: int = 3, min_delta: float = 1e-3, max_epochs: int | None = None) -> ConvergenceResult:
    """Drive fit_to_convergence with a scripted val-loss sequence (a dummy param module, no-op steps)."""
    module = nn.Linear(1, 1)  # dummy: gives fit_to_convergence a real state_dict to snapshot
    seq = iter(vals)
    state = {"last": vals[0] if vals else 0.0}

    def step() -> None:
        pass  # no real training; the scripted sequence stands in for the held-out-loss evolution

    def val() -> float:
        state["last"] = next(seq, state["last"])  # after the sequence is exhausted, hold flat
        return state["last"]

    return fit_to_convergence(module, step, val, max_epochs=max_epochs or len(vals), check_every=check_every, patience=patience, min_delta=min_delta)


def run_selftest() -> bool:
    """Known-answer checks: flattening → converged; monotone-still-falling → hit_cap; slow-creep flagged."""
    ok = True

    # (a) decreases then flattens → converged=True, hit_cap=False. The 0.24→0.2399 drop (0.0001) is
    #     below min_delta=1e-3 so it does NOT count as improvement → best correctly stays at 0.24.
    r = _scripted_run([1.0, 0.5, 0.3, 0.25, 0.24, 0.2400, 0.2399, 0.2399, 0.2399, 0.2399], patience=3, min_delta=1e-3)
    ok_a = r.converged and not r.hit_cap and abs(r.best_val - 0.24) < _TEST_TOL
    print(f"[convergence selftest] (a) flatten→converged: converged={r.converged} hit_cap={r.hit_cap} best={r.best_val:.4f}  {'PASS' if ok_a else 'FAIL'}")
    ok = ok and ok_a

    # (b) strictly falling the whole time, capped early → hit_cap=True, converged=False, still_improving=True.
    r = _scripted_run([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], patience=3, min_delta=1e-3, max_epochs=10)
    ok_b = (not r.converged) and r.hit_cap and r.still_improving
    print(f"[convergence selftest] (b) still-falling→hit_cap: converged={r.converged} hit_cap={r.hit_cap} still_improving={r.still_improving}  {'PASS' if ok_b else 'FAIL'}")
    ok = ok and ok_b

    # (c) best-weights (not last): improve, then WORSEN → best_val stays at the good value, best_epoch < stop.
    r = _scripted_run([1.0, 0.30, 0.29, 0.50, 0.60, 0.70], patience=3, min_delta=1e-3)
    ok_c = abs(r.best_val - 0.29) < _TEST_TOL and r.best_epoch < r.stop_epoch
    print(f"[convergence selftest] (c) keeps best-not-last: best={r.best_val:.4f}@{r.best_epoch} stop@{r.stop_epoch}  {'PASS' if ok_c else 'FAIL'}")
    ok = ok and ok_c

    # (d) explodes then plateaus high (patience-stops without recovering) → diverged=True, trustworthy=False.
    #     Mirrors the real Z120 seed-1 shared_readout case: best CE 1.339 -> final 3.032.
    r = _scripted_run([1.0, 0.5, 0.3, 0.25, 0.24, 5.0, 4.9, 4.8], patience=3, min_delta=1e-3)
    ok_d = r.diverged and not r.trustworthy
    print(f"[convergence selftest] (d) spike-plateau→diverged: diverged={r.diverged} trustworthy={r.trustworthy} best={r.best_val:.4f}  {'PASS' if ok_d else 'FAIL'}")
    ok = ok and ok_d

    # (e) healthy near-zero trajectory: harmless wobble off a tiny best must NOT trip the ratio term —
    #     this is exactly why the absolute floor (DIVERGENCE_ABS_EPS) is mandatory.
    r = _scripted_run([0.02, 0.01, 0.005, 0.003, 0.0015, 0.002, 0.003, 0.0045], patience=3, min_delta=1e-4)
    ok_e = not r.diverged and abs(r.best_val - 0.0015) < _TEST_TOL
    print(f"[convergence selftest] (e) near-zero wobble→not diverged: diverged={r.diverged} best={r.best_val:.4f} final={r.trajectory[-1][1]:.4f}  {'PASS' if ok_e else 'FAIL'}")
    ok = ok and ok_e

    print(f"[convergence selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Runs the selftest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Scripted known-answer checks of the convergence gate.")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
