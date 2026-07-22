"""WSEL-16 frozen-bias diagnostic (root, 2026-07-22) — run BEFORE interpreting A_STOPGRAD's accuracy failure.

The worker-flagged interpretation caveat (width.md WSEL-16 STATUS block): under the stop-gradient
loss the shared readout bias `mean_head.bias` appears ONLY inside the detached term
(`width_wsel16.stopgrad_all_widths_pred`, `s_prev.detach()`), so it receives zero gradient at every
width and stays frozen at its init value for the whole run. Question this script answers with
numbers: does that frozen scalar EXPLAIN the ~15-20x full-width accuracy failure, or is the failure
attributable to greedy training itself?

Method, per tier-1 seed (0/1/2):
1. Rebuild the cell's data EXACTLY as `width_wsel16.run_cell` does (same carves, same norm).
2. Rebuild the init state (`torch.manual_seed(seed)` then construct) and assert the saved
   `mean_head.bias` equals the init bias bit-for-bit — proving "frozen at init" from disk.
3. Baseline: reproduce the landed full-width held-out MSE from the saved state (cross-check vs
   `frozen.json.full_width_held_out_mse_by_seed`).
4. Bias-only repair: closed-form refit of the ONE scalar on the phase-1 TRAIN split
   (`b_opt = b_frozen + mean(y_tr - S_12(x_tr))`), everything else untouched; re-read the
   full-width held-out MSE. The repair is a 1-dof post-hoc fit — if the failure were the bias,
   this closes it; a scalar cannot fake capacity.

Output: `frozen_bias_diag.json` beside this script. No training, no state file is modified.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from automl_package.examples import width_wsel16 as w16

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
W_MAX = 12
REPRODUCTION_REL_TOL = 1e-6  # float32 eval on identical tensors — anything beyond this is a data/protocol mismatch, not noise
B_HEADS_FULL_WIDTH_VAL_MSE = {0: 0.024037670344114304, 1: 0.030796518549323082, 2: 0.028489189222455025}  # frozen.json, tier 1


def _tier1_data(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicates `width_wsel16.run_cell`'s tier-1 data prep verbatim (train split + val carve, standardized)."""
    cfg = w16._TIER_CONFIG[w16.Tier.ONE]
    x_tr, y_tr, _region = w16.nwn.make_hetero(cfg.n_train, seed, sigma=cfg.sigma)
    p1_idx = np.arange(0, cfg.n_train, 2)
    x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
    val_mask = (np.arange(len(x_p1)) % w16.cwe.VAL_EVERY) == 0
    norm = w16.cwe._standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
    x_tr_t, y_tr_t = w16.cwe._to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
    x_val_t, y_val_t = w16.cwe._to_std_tensors(x_p1[val_mask], y_p1[val_mask], norm)
    return x_tr_t, y_tr_t, x_val_t, y_val_t


def main() -> None:
    """Runs the three-seed diagnostic and writes `frozen_bias_diag.json` (see module docstring)."""
    out: dict = {"task": "WSEL-16 frozen-bias diagnostic", "arm": "a_stopgrad", "tier": 1, "per_seed": {}}
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        net_init = w16._build_net(w16.Arm.A_STOPGRAD, W_MAX)
        bias_init = float(net_init.mean_head.bias.detach().item())

        net = w16._build_net(w16.Arm.A_STOPGRAD, W_MAX)
        state_path = os.path.join(RESULTS_DIR, f"state_a_stopgrad_tier1_seed{seed}.pt")
        net.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
        net.eval()
        bias_saved = float(net.mean_head.bias.detach().item())

        x_tr_t, y_tr_t, x_val_t, y_val_t = _tier1_data(seed)
        with torch.no_grad():
            mse_before = float(w16.nwn._width_mse(net, W_MAX, x_val_t, y_val_t).item())
            mean_tr, _ = net.forward_width(x_tr_t, W_MAX)
            residual_mean = float((y_tr_t - mean_tr.squeeze(1)).mean().item())
            net.mean_head.bias += residual_mean  # the 1-dof closed-form repair
            mse_after = float(w16.nwn._width_mse(net, W_MAX, x_val_t, y_val_t).item())
            mse_after_by_width = {k: float(w16.nwn._width_mse(net, k, x_val_t, y_val_t).item()) for k in range(1, W_MAX + 1)}

        out["per_seed"][str(seed)] = {
            "bias_init": bias_init,
            "bias_saved": bias_saved,
            "frozen_at_init": bias_saved == bias_init,
            "bias_repair_delta": residual_mean,
            "full_width_val_mse_landed_reference": None,  # filled by cross-check below
            "full_width_val_mse_before_repair": mse_before,
            "full_width_val_mse_after_repair": mse_after,
            "b_heads_full_width_val_mse": B_HEADS_FULL_WIDTH_VAL_MSE[seed],
            "rel_gap_to_b_heads_before": mse_before / B_HEADS_FULL_WIDTH_VAL_MSE[seed] - 1.0,
            "rel_gap_to_b_heads_after": mse_after / B_HEADS_FULL_WIDTH_VAL_MSE[seed] - 1.0,
            "val_mse_after_repair_by_width": {str(k): v for k, v in mse_after_by_width.items()},
        }
        with open(os.path.join(RESULTS_DIR, f"wsel16_a_stopgrad_tier1_seed{seed}.json")) as f:
            cell = json.load(f)
        landed = cell["full_width_held_out_mse"]
        rec = out["per_seed"][str(seed)]
        rec["full_width_val_mse_landed_reference"] = landed
        rec["reproduces_landed_value"] = abs(rec["full_width_val_mse_before_repair"] - landed) / landed < REPRODUCTION_REL_TOL

    frac_closed = [
        1.0 - (r["full_width_val_mse_after_repair"] - r["b_heads_full_width_val_mse"]) / (r["full_width_val_mse_before_repair"] - r["b_heads_full_width_val_mse"])
        for r in out["per_seed"].values()
    ]
    out["summary"] = {
        "all_seeds_frozen_at_init": all(r["frozen_at_init"] for r in out["per_seed"].values()),
        "all_seeds_reproduce_landed_value": all(r["reproduces_landed_value"] for r in out["per_seed"].values()),
        "fraction_of_gap_to_b_heads_closed_by_bias_repair": {str(s): f for s, f in zip(out["per_seed"], frac_closed, strict=True)},
        "reading": (
            "If fraction_of_gap... is near 0 the frozen bias does NOT explain the accuracy failure "
            "(a scalar offset cannot fake capacity) and greedy training remains the live explanation; "
            "near 1 means the failure is an artifact of the spec-literal stop-grad implementation."
        ),
    }
    out_path = os.path.join(RESULTS_DIR, "frozen_bias_diag.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))
    for seed, r in out["per_seed"].items():
        print(
            f"seed {seed}: frozen_at_init={r['frozen_at_init']} repro={r['reproduces_landed_value']} "
            f"mse {r['full_width_val_mse_before_repair']:.4f} -> {r['full_width_val_mse_after_repair']:.4f} "
            f"(B_HEADS {r['b_heads_full_width_val_mse']:.4f}; bias delta {r['bias_repair_delta']:+.4f})"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
