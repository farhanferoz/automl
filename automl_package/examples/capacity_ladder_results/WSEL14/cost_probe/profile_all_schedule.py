"""Cost-attribution profiler for the WSEL-14 all-widths-vs-uniform-b1 anomaly.

MEASUREMENT ONLY -- imports the real driver / architecture / net code and reuses it unmodified.
The only "new" code here is (a) the tiny data-setup glue copied verbatim from
`kdropout_converged_width_experiment.run_case` (that glue is inline in `run_case`, not a
separate function, so it cannot be imported) and (b) a timing wrapper placed around the net's
own `hidden` bound method so trunk-forward time can be separated from the per-width head loop
without changing any tensor math. Every actual computation (`_sampled_widths_total_loss`,
`.backward()`, `opt.step()`, the periodic `_width_loss` eval) calls the SAME functions the
production trainer calls.

Canonical cell (fixed, matches the task brief / WSEL-14 frozen.json config):
    arch=shared_trunk, loss=mse, toy=hetero, n_train=1500, sigma=0.05, w_max=12, lr=1e-2, seed=0
    check_every=500 (convergence.DEFAULT_CHECK_EVERY) -- convergence gating itself is NOT run
    (no ConvergenceTracker, no early stop): this script always runs exactly --steps steps.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python profile_all_schedule.py \
        --order A --steps 2000 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_REPO_ROOT = "/home/ff235/dev/MLResearch/automl"
_EXAMPLES_DIR = os.path.join(_REPO_ROOT, "automl_package", "examples")
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, _REPO_ROOT)

import converged_width_experiment as cwe  # noqa: E402
import kdropout_converged_width_experiment as kexp  # noqa: E402
import nested_width_net as nwn  # noqa: E402

# Canonical cell, fixed by the task brief -- not CLI knobs, so a stray flag can't drift the cell.
SEED = 0
W_MAX = 12
N_TRAIN = 1500
SIGMA = nwn.HETERO_NOISE_SIGMA  # 0.05
LOSS = kexp.LossType.MSE
CHECK_EVERY = 500  # convergence.DEFAULT_CHECK_EVERY -- production's real cadence, gating itself skipped.
UNIFORM_DRAW_N_B1 = 1  # "b1" in the frozen reference == uniform draw-1.

# Order of {schedule_label: nwn.WidthSchedule} run within one invocation. Two fixed orders (A, B)
# let the caller check whether a schedule's timing depends on where in the process it runs
# (thread-pool / allocator warm-up landing on whichever schedule goes first).
_SCHEDULE_MAP = {
    "b1": nwn.WidthSchedule.UNIFORM,
    "sandwich": nwn.WidthSchedule.SANDWICH,
    "all": nwn.WidthSchedule.ALL,
}
_ORDERS = {
    "A": ["b1", "sandwich", "all"],
    "B": ["all", "sandwich", "b1"],
}


def _build_data(seed: int, n_train: int, sigma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verbatim copy of `run_case`'s HETERO data-setup glue (kdropout_converged_width_experiment.py:368-386).

    Not importable as a function -- it is inline in `run_case` -- so it is reproduced here exactly
    (same calls to `nwn.make_hetero` / `cwe._standardize_fit` / `cwe._to_std_tensors`, same
    indexing) rather than re-derived. Test set (x_te/y_te) is dropped: this script never scores
    bars, only trains -- it does not belong in this task's write set to begin with.
    """
    x_tr, y_tr, _reg_tr = nwn.make_hetero(n_train, seed, sigma=sigma)
    p1_idx = np.arange(0, n_train, 2)
    x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
    val_mask = (np.arange(len(x_p1)) % cwe.VAL_EVERY) == 0
    norm = cwe._standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
    x_tr_t, y_tr_t = cwe._to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
    x_val_t, y_val_t = cwe._to_std_tensors(x_p1[val_mask], y_p1[val_mask], norm)
    return x_tr_t, y_tr_t, x_val_t, y_val_t


def _select_widths(schedule: nwn.WidthSchedule, w_max: int, gen: torch.Generator, mid_candidates: list[int], n_mid_draw: int, uniform_draw_n: int) -> list[int]:
    """Verbatim copy of the width-selection branch inside `_train_kdropout_to_convergence`
    (kdropout_converged_width_experiment.py:289-300) -- reproduced exactly, not re-derived,
    because that branch lives inside the epoch loop and is not its own function to import.
    """
    if schedule is nwn.WidthSchedule.UNIFORM:
        return torch.randint(1, w_max + 1, (uniform_draw_n,), generator=gen).tolist()
    if schedule is nwn.WidthSchedule.ALL:
        return list(range(1, w_max + 1))
    widths = [1, w_max]
    if n_mid_draw:
        perm = torch.randperm(len(mid_candidates), generator=gen)[:n_mid_draw]
        widths += [mid_candidates[i] for i in perm.tolist()]
    return widths


def run_one_schedule(label: str, schedule: nwn.WidthSchedule, steps: int, warmup: int) -> dict:
    """Runs `steps` training steps of `_train_kdropout_to_convergence`'s real inner loop body,
    with phase timers added around: width-selection (RNG), trunk forward (net.hidden), the
    per-width head-forward+loss loop, backward(), opt.step(), and the periodic convergence-style
    eval block (every CHECK_EVERY steps) -- using the SAME functions production calls throughout.
    No ConvergenceTracker / early stop: always runs exactly `steps` steps (explicit non-goal:
    no full convergence runs).
    """
    x_tr_t, y_tr_t, x_val_t, y_val_t = _build_data(SEED, N_TRAIN, SIGMA)

    torch.manual_seed(SEED)
    net = nwn.SharedTrunkPerWidthHeadNet(w_max=W_MAX)  # arch=shared_trunk, exactly as run_case (line 393-394)
    opt = torch.optim.Adam(net.parameters(), lr=cwe.LR)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(SEED))
    mid_candidates = list(range(2, W_MAX))
    n_mid_draw = min(2, len(mid_candidates))

    # Timing wrapper around net.hidden -- the ONLY monkeypatch in this script. Everything downstream
    # of `self.hidden(x)` inside `sampled_widths_forward` / `forward_width` is untouched; this only
    # measures the wall time of that one bound method's real calls.
    orig_hidden = net.hidden
    _hidden_accum = {"s": 0.0}

    def _timed_hidden(x):
        t0 = time.perf_counter()
        out = orig_hidden(x)
        _hidden_accum["s"] += time.perf_counter() - t0
        return out

    net.hidden = _timed_hidden

    phases = {"widths_select": [], "trunk_forward": [], "head_loop": [], "backward": [], "opt_step": []}
    eval_block_times = []
    net.train()
    t_loop0 = time.perf_counter()
    for epoch in range(1, steps + 1):
        opt.zero_grad()

        t0 = time.perf_counter()
        widths = _select_widths(schedule, W_MAX, gen, mid_candidates, n_mid_draw, UNIFORM_DRAW_N_B1)
        t_select = time.perf_counter() - t0

        _hidden_accum["s"] = 0.0
        t0 = time.perf_counter()
        total_loss = kexp._sampled_widths_total_loss(LOSS, net, widths, x_tr_t, y_tr_t)
        t_fwdloss = time.perf_counter() - t0
        t_trunk = _hidden_accum["s"]
        t_head_loop = t_fwdloss - t_trunk

        t0 = time.perf_counter()
        total_loss.backward()
        t_backward = time.perf_counter() - t0

        t0 = time.perf_counter()
        opt.step()
        t_opt = time.perf_counter() - t0

        if epoch > warmup:
            phases["widths_select"].append(t_select)
            phases["trunk_forward"].append(t_trunk)
            phases["head_loop"].append(t_head_loop)
            phases["backward"].append(t_backward)
            phases["opt_step"].append(t_opt)

        if epoch % CHECK_EVERY == 0:
            net.eval()
            _hidden_accum["s"] = 0.0
            t0 = time.perf_counter()
            with torch.no_grad():
                per_width_val = {k: float(kexp._width_loss(LOSS, net, k, x_val_t, y_val_t).item()) for k in range(1, W_MAX + 1)}
            t_eval = time.perf_counter() - t0
            net.train()
            del per_width_val
            if epoch > warmup:
                eval_block_times.append(t_eval)

    total_wall_s = time.perf_counter() - t_loop0
    net.hidden = orig_hidden  # restore, though the net is discarded right after

    n_counted = steps - warmup
    phase_means_ms = {k: (sum(v) / len(v) * 1000.0 if v else 0.0) for k, v in phases.items()}
    phase_sum_ms_per_step = sum(phase_means_ms.values())
    eval_total_s = sum(eval_block_times)
    eval_amortized_ms_per_step = (eval_total_s / n_counted * 1000.0) if n_counted else 0.0

    return {
        "label": label,
        "schedule": schedule.value,
        "steps": steps,
        "warmup_excluded": warmup,
        "n_counted": n_counted,
        "total_wall_clock_s": total_wall_s,
        "total_wall_clock_ms_per_step": total_wall_s / steps * 1000.0,
        "phase_means_ms_per_step": phase_means_ms,
        "phase_sum_ms_per_step_excl_eval": phase_sum_ms_per_step,
        "n_eval_blocks": len(eval_block_times),
        "eval_block_mean_ms": (eval_total_s / len(eval_block_times) * 1000.0) if eval_block_times else 0.0,
        "eval_amortized_ms_per_step": eval_amortized_ms_per_step,
        "phase_sum_plus_eval_ms_per_step": phase_sum_ms_per_step + eval_amortized_ms_per_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", choices=["A", "B"], required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=50, help="Steps excluded from the per-step stats (still executed).")
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    order = _ORDERS[args.order]
    print(f"[costprobe] order={args.order} sequence={order} steps={args.steps} warmup={args.warmup}", flush=True)

    for label in order:
        schedule = _SCHEDULE_MAP[label]
        t0 = time.perf_counter()
        result = run_one_schedule(label, schedule, args.steps, args.warmup)
        wall = time.perf_counter() - t0
        result["order"] = args.order
        result["invocation_wall_s"] = wall
        out_path = os.path.join(args.out_dir, f"result_order{args.order}_{label}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"[costprobe] order={args.order} label={label:8s} total_ms_per_step={result['total_wall_clock_ms_per_step']:.3f} "
            f"phase_sum_excl_eval={result['phase_sum_ms_per_step_excl_eval']:.3f} eval_amortized={result['eval_amortized_ms_per_step']:.3f} "
            f"wall_s={wall:.2f} -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
