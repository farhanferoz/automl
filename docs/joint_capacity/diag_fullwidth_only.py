"""Diagnostic: does the joint substrate fit S1 at FULL width when the loss is NOT diluted across the
width grid? Trains JointCapacityNet (readout) with the CE loss on the w=64 cells ONLY (all T), same
data/seed as the failed J-1 pilot. Isolates grid-loss/block-corruption (hypothesis a) from intrinsic
multi-track-fold difficulty (hypothesis b). Reversible; scratchpad-only; no toy/plan files touched.

Contrast anchor: the failed pilot trained on all 20 (w,T) cells -> S1 per-track 0.58-0.79 (fails 0.90).
Read: if this full-width-only run reaches >=0.90 per (A,t*) cell -> grid corruption (a). If it stays
~0.78 -> multi-track fold (b).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/ff235/dev/MLResearch/automl/automl_package/examples")

import convergence as cvg  # noqa: E402
import depth_selection_toy as dst  # noqa: E402
import joint_capacity_toy as jct  # noqa: E402

DEVICE = os.environ.get("AUTOML_DEVICE", "cpu")
SEED = 0
N_PER_CELL = 1000  # match the failed pilot exactly -> the ONLY changed variable is the loss support

data = jct.build_joint_dataset(n_per_cell=N_PER_CELL, seed=SEED)
x_tr, _am_tr, pp_tr, _a_tr, _t_tr = jct._pool_cells(data, "tr")
x_val, am_val, pp_val, a_val, t_val = jct._pool_cells(data, "val")

torch.manual_seed(SEED)
net = jct.JointCapacityNet(jct.WidthMode.READOUT, data["n_gen_per_step"], data["n_classes"], k_max=data["k_max"]).to(DEVICE)
w_max_idx = len(jct.WIDTH_LADDER) - 1
T_LADDER = jct.T_LADDER

x_tr_t = torch.as_tensor(x_tr, dtype=torch.float32, device=DEVICE)
x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=DEVICE)
y_tr = {t: torch.as_tensor(pp_tr[:, :, t - 1], dtype=torch.long, device=DEVICE) for t in T_LADDER}
y_val = {t: torch.as_tensor(pp_val[:, :, t - 1], dtype=torch.long, device=DEVICE) for t in T_LADDER}
opt = torch.optim.Adam(net.parameters(), lr=jct.LR_DEFAULT)
ce = nn.CrossEntropyLoss()


def _loss(x: torch.Tensor, y: dict) -> torch.Tensor:
    g = net.forward_grid(x, t_ladder=T_LADDER, width_ladder_idx=(w_max_idx,))  # full width only
    return sum(ce(g[(w_max_idx, t)].reshape(-1, net.n_classes), y[t].reshape(-1)) for t in T_LADDER) / len(T_LADDER)


def step_fn() -> None:
    opt.zero_grad()
    loss = _loss(x_tr_t, y_tr)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), jct.GRAD_CLIP_MAX_NORM)
    opt.step()


def val_fn() -> float:
    net.eval()
    with torch.no_grad():
        v = float(_loss(x_val_t, y_val).item())
    net.train()
    return v


res = cvg.fit_to_convergence(net, step_fn, val_fn, max_epochs=jct.MAX_EPOCHS_DEFAULT, check_every=dst.CHECK_EVERY, patience=dst.PATIENCE, min_delta=dst.MIN_DELTA)

net.eval()
with torch.no_grad():
    g = net.forward_grid(x_val_t, t_ladder=(jct.L,), width_ladder_idx=(w_max_idx,))
    pred = g[(w_max_idx, jct.L)].argmax(-1).cpu().numpy()
full_label = pp_val[:, :, -1]
correct = pred == full_label

print(f"[diag full-width-only] converged={res.converged} diverged={res.diverged} hit_cap={res.hit_cap} trustworthy={res.trustworthy}")
print("[diag] S1 per (A,t*) — per-track active acc @ w=64,T=10 (contrast: pilot grid-loss was 0.58-0.79):")
worst = 1.0
for a in range(1, jct.K_MAX + 1):
    for t in jct.T_STAR_LADDER:
        m = (a_val == a) & (t_val == t)
        entries = correct[m][am_val[m]]
        acc = float(entries.mean()) if entries.size else float("nan")
        worst = min(worst, acc)
        print(f"   a={a},t={t}: {acc:.3f}")
s1_pass = worst >= jct.S1_FIT_ACC
print(f"[diag] worst cell = {worst:.3f}  ->  S1>={jct.S1_FIT_ACC} all cells: {s1_pass}")
print(f"[diag] VERDICT: {'grid-corruption (hyp a) -- full width fits alone' if s1_pass else 'multi-track fold bottleneck (hyp b) -- dedicated full width still underfits'}")
