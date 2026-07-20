"""Thin re-export shim — the convergence-gating logic now lives in `automl_package.utils.convergence`.

Kept here (rather than deleted) so every existing example script's `import convergence as cvg` /
`from automl_package.examples.convergence import ...` keeps resolving unchanged. Also keeps the
scripted selftest CLI, since that is example-script behavior, not package logic.

Selftest:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/convergence.py --selftest
"""

from __future__ import annotations

import argparse
import os
import sys

import torch.nn as nn

# Repo root, so `import automl_package` works when this file is run directly as a script
# (`python automl_package/examples/convergence.py`), not just via `-m` or an installed package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from automl_package.utils.convergence import (  # noqa: E402
    DEFAULT_CHECK_EVERY,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MIN_DELTA,
    DEFAULT_PATIENCE,
    DIVERGENCE_ABS_EPS,
    DIVERGENCE_REL_FACTOR,
    ConvergenceResult,
    ConvergenceTracker,
    fit_to_convergence,
    format_trajectory,
)

__all__ = [
    "DEFAULT_CHECK_EVERY",
    "DEFAULT_MAX_EPOCHS",
    "DEFAULT_MIN_DELTA",
    "DEFAULT_PATIENCE",
    "DIVERGENCE_ABS_EPS",
    "DIVERGENCE_REL_FACTOR",
    "ConvergenceResult",
    "ConvergenceTracker",
    "fit_to_convergence",
    "format_trajectory",
]

_TEST_TOL = 1e-6  # selftest float-compare tolerance


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
