"""WSEL-18 acceptance benchmark: fused ALL-schedule per-step wall-clock vs per-head SANDWICH per-step.

`docs/plans/capacity_programme/width.md` WSEL-18 spec (4), on the canonical cell. Modelled on `capacity_ladder_results/WSEL14/cost_probe/profile_all_schedule.py`'s cost-probe shape
(same canonical cell, same "reuse the real driver's inner-loop body verbatim" method, same
warmup-then-count convention) but narrower: this only needs the two numbers the spec asks for --
fused/ALL and per-head/SANDWICH total per-step wall-clock -- not that probe's five-way phase
breakdown, so there is no `net.hidden` monkeypatch here. Per-step timing only, a BOUNDED number of
steps (`--steps`, default 2000 -- the spec's own suggestion): this never trains to convergence
(non-goal).

This is the measurement that verifies the "dominance, no decision" premise: ALL trains every width
(12) every step, SANDWICH trains only 4 -- so ALL doing MORE work per step and still landing at or
below SANDWICH's per-step wall-clock (once fused) is the point being measured, not an apples-to-
apples widths-per-step comparison.

Canonical cell (fixed, matches WSEL-14's cost_probe cell and the WSEL-18 spec's own wording):
    arch=shared_trunk, loss=mse, toy=hetero, n_train=1500, sigma=0.05, w_max=12, lr=1e-2, seed=0

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python \
        automl_package/examples/capacity_ladder_results/WSEL18/bench.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../capacity_ladder_results/WSEL18
_EXAMPLES_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))  # .../automl_package/examples
_REPO_ROOT = os.path.dirname(os.path.dirname(_EXAMPLES_DIR))  # repo root
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, _REPO_ROOT)

import converged_width_experiment as cwe  # noqa: E402
import kdropout_converged_width_experiment as kexp  # noqa: E402
import nested_width_net as nwn  # noqa: E402

from automl_package.utils.pytorch_utils import get_device  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

# Canonical cell, fixed -- not CLI knobs, so a stray flag can't drift the cell (WSEL-14 cost_probe precedent).
SEED = 0
W_MAX = 12
N_TRAIN = 1500
SIGMA = nwn.HETERO_NOISE_SIGMA  # 0.05
LOSS = kexp.LossType.MSE
STEPS = 2000
WARMUP = 50
OUT_PATH = os.path.join(_THIS_DIR, "bench.json")


def _build_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Verbatim copy of `run_case`'s HETERO train-split glue.

    `kdropout_converged_width_experiment.py` ~399-413 -- not importable as a function (it is inline
    in `run_case`), and this script never scores bars or needs the held-out split, only trains, so
    only the train tensors are built.
    """
    x_tr, y_tr, _reg_tr = nwn.make_hetero(N_TRAIN, SEED, sigma=SIGMA)
    p1_idx = np.arange(0, N_TRAIN, 2)
    x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
    val_mask = (np.arange(len(x_p1)) % cwe.VAL_EVERY) == 0
    norm = cwe._standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
    x_tr_t, y_tr_t = cwe._to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
    return x_tr_t, y_tr_t


def run_one(*, fused_heads: bool, schedule: nwn.WidthSchedule, steps: int, warmup: int) -> dict:
    """Runs `steps` real training steps and times each step's total wall clock.

    Each step is `_sampled_widths_total_loss` -> `.backward()` -> `opt.step()`, the SAME functions
    `_train_kdropout_to_convergence` calls.
    """
    x_tr_t, y_tr_t = _build_data()

    torch.manual_seed(SEED)
    net = nwn.SharedTrunkPerWidthHeadNet(w_max=W_MAX, fused_heads=fused_heads)  # arch=shared_trunk, exactly as run_case
    opt = torch.optim.Adam(net.parameters(), lr=cwe.LR)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(SEED))
    mid_candidates = list(range(2, W_MAX))
    n_mid_draw = min(2, len(mid_candidates))

    step_times = []
    net.train()
    t_loop0 = time.perf_counter()
    for _epoch in range(1, steps + 1):
        opt.zero_grad()
        if schedule is nwn.WidthSchedule.ALL:
            widths = list(range(1, W_MAX + 1))
        else:  # SANDWICH -- the only two schedules this benchmark needs (kdropout_converged_width_experiment.py:293-300)
            widths = [1, W_MAX]
            if n_mid_draw:
                perm = torch.randperm(len(mid_candidates), generator=gen)[:n_mid_draw]
                widths += [mid_candidates[i] for i in perm.tolist()]

        t0 = time.perf_counter()
        total_loss = kexp._sampled_widths_total_loss(LOSS, net, widths, x_tr_t, y_tr_t)
        total_loss.backward()
        opt.step()
        step_times.append(time.perf_counter() - t0)

    total_wall_s = time.perf_counter() - t_loop0
    counted = step_times[warmup:]
    n_counted = len(counted)
    return {
        "fused_heads": fused_heads,
        "schedule": schedule.value,
        "steps": steps,
        "warmup_excluded": warmup,
        "n_counted": n_counted,
        "total_wall_clock_s": total_wall_s,
        "mean_step_wall_clock_ms": (sum(counted) / n_counted * 1000.0) if n_counted else 0.0,
    }


def main() -> None:
    """Runs both configs (fused/ALL, per-head/SANDWICH) and writes `bench.json` next to this script."""
    print(f"[WSEL18-bench] arch=shared_trunk loss={LOSS.value} toy=hetero n_train={N_TRAIN} sigma={SIGMA:g} w_max={W_MAX} steps={STEPS} warmup={WARMUP}", flush=True)

    fused_all = run_one(fused_heads=True, schedule=nwn.WidthSchedule.ALL, steps=STEPS, warmup=WARMUP)
    print(f"[WSEL18-bench] fused/ALL       mean_step_ms={fused_all['mean_step_wall_clock_ms']:.4f} total_s={fused_all['total_wall_clock_s']:.3f}", flush=True)

    per_head_sandwich = run_one(fused_heads=False, schedule=nwn.WidthSchedule.SANDWICH, steps=STEPS, warmup=WARMUP)
    print(
        f"[WSEL18-bench] per_head/SANDWICH mean_step_ms={per_head_sandwich['mean_step_wall_clock_ms']:.4f} total_s={per_head_sandwich['total_wall_clock_s']:.3f}",
        flush=True,
    )

    speedup = per_head_sandwich["mean_step_wall_clock_ms"] / fused_all["mean_step_wall_clock_ms"]
    result = {
        "fused_all": fused_all,
        "per_head_sandwich": per_head_sandwich,
        "speedup_fused_all_over_per_head_sandwich": speedup,
        "config": {
            "arch": "shared_trunk",
            "loss": LOSS.value,
            "toy": "hetero",
            "w_max": W_MAX,
            "n_train": N_TRAIN,
            "sigma": SIGMA,
            "lr": cwe.LR,
            "seed": SEED,
            "device": str(get_device()),
        },
        "provenance": run_provenance(),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[WSEL18-bench] speedup(fused ALL vs per-head SANDWICH)={speedup:.3f}x -> wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
