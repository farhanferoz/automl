"""Root pre-grid check: does WSEL-16's `_train_custom_to_convergence` reproduce
`kce._train_kdropout_to_convergence` bit-for-bit on a cell BOTH can express?

Why: stage-1's PRIMARY bar compares A_STOPGRAD (custom trainer) against B_HEADS (real trainer)
on tier 1. If the two loops diverge (draw sequence, optimizer, tracker cadence, checkpointing),
that comparison carries a trainer confound. This proves parity on B_HEADS tier-1 semantics:
same init, same tensors, same seed -> expect identical convergence records and final weights.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ff235/dev/MLResearch/automl/automl_package/examples")
os.environ.setdefault("AUTOML_DEVICE", "cpu")

import converged_width_experiment as cwe  # noqa: E402
import kdropout_converged_width_experiment as kce  # noqa: E402
import nested_width_net as nwn  # noqa: E402
import width_wsel16 as w16  # noqa: E402

W_MAX = 12
N_TRAIN = 400
SEED = 0
FLOAT_NOISE_TOL = 1e-9  # below this, differences are float noise, not a diverging loop
KW = dict(max_epochs=20000, check_every=500, patience=6, min_delta=0.002)

# Data prep mirrors width_wsel16.run_cell tier-1 exactly.
x_tr, y_tr, region_tr = nwn.make_hetero(N_TRAIN, SEED, sigma=0.05)
p1_idx = np.arange(0, N_TRAIN, 2)
x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
region_p1 = region_tr[p1_idx]
val_mask = (np.arange(len(x_p1)) % cwe.VAL_EVERY) == 0
norm = cwe._standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
x_tr_t, y_tr_t = cwe._to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
x_val_t, y_val_t = cwe._to_std_tensors(x_p1[val_mask], y_p1[val_mask], norm)
region_split = region_p1[~val_mask]
sigma_tr = w16._sigma_true_tensor(nwn.Toy.HETERO, region_split, 0.05, norm)
sigma_val = w16._sigma_true_tensor(nwn.Toy.HETERO, region_p1[val_mask], 0.05, norm)

# Net A: the real trainer.
torch.manual_seed(SEED)
net_a = nwn.SharedTrunkPerWidthHeadNet(w_max=W_MAX)
conv_a, _ = kce._train_kdropout_to_convergence(
    net_a, x_tr_t, y_tr_t, x_val_t, y_val_t,
    arch=kce.Arch.SHARED_TRUNK, loss=kce.LossType.MSE, seed=SEED,
    schedule=nwn.WidthSchedule.SANDWICH, **KW,
)

# Net B: the custom trainer, same init, same everything.
torch.manual_seed(SEED)
net_b = nwn.SharedTrunkPerWidthHeadNet(w_max=W_MAX)
conv_b = w16._train_custom_to_convergence(
    net_b, x_tr_t, y_tr_t, x_val_t, y_val_t,
    total_loss_fn=w16._make_standard_total_loss_fn(w16.Tier.ONE, sigma_tr),
    per_width_val_fn=w16._make_per_width_val_fn(w16.Tier.ONE, sigma_val),
    w_max=W_MAX, seed=SEED, independent_checkpointing=False, **KW,
)

rows, worst = [], 0.0
for k in range(1, W_MAX + 1):
    ra, rb = conv_a[k], conv_b[k]
    d = abs(ra.best_val - rb.best_val)
    worst = max(worst, d)
    rows.append((k, ra.stop_epoch, rb.stop_epoch, ra.best_val, rb.best_val, d))
print(f"{'k':>3} {'stopA':>7} {'stopB':>7} {'bestA':>12} {'bestB':>12} {'|diff|':>10}")
for k, sa, sb, ba, bb, d in rows:
    print(f"{k:>3} {sa:>7} {sb:>7} {ba:>12.8f} {bb:>12.8f} {d:>10.2e}")

sd_a, sd_b = net_a.state_dict(), net_b.state_dict()
wdiff = max((sd_a[key] - sd_b[key]).abs().max().item() for key in sd_a)
stop_match = all(conv_a[k].stop_epoch == conv_b[k].stop_epoch for k in range(1, W_MAX + 1))
print(f"\nmax weight |diff| = {wdiff:.3e}   max best_val |diff| = {worst:.3e}   stop_epochs_match = {stop_match}")
if wdiff == 0.0 and worst == 0.0 and stop_match:
    verdict = "PARITY"
elif wdiff < FLOAT_NOISE_TOL and worst < FLOAT_NOISE_TOL:
    verdict = "NEAR (float-noise?)"
else:
    verdict = "DIVERGENT"
print(f"VERDICT: {verdict}")
