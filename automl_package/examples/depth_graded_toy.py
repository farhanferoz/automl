"""Graded per-input-depth toy — ONE shared recurrent block serving variable per-input DEPTH (`depth.md` D1b).

Trains one shared block on a MIX of S5 word lengths and reads held-out accuracy PER LENGTH, at the
canonical LR 3e-3 (the depth edition of the width toy's #2 `SharedTrunkPerWidthHeadNet`).

**What this adds over `depth_composition_toy.py`** (which established the single-length make-or-break:
a length-n S5 recurrent net beats a wide-shallow MLP, gap widening with n). This module trains ONE
shared block on a MIX of word lengths and reads held-out accuracy PER LENGTH — the actual "one
network serves per-input variable depth" charter question, the depth edition of the width toy's #2
`SharedTrunkPerWidthHeadNet`. The per-input "depth" of a length-ℓ word = its ℓ recurrent unroll steps.

**Why LR 3e-3 (not the composition toy's 1e-2).** The n=10 trainability spike proved the length-10
"wall" (val 0.569 at LR 1e-2) was an OPTIMIZATION artifact, not representational: LR 3e-3 → 0.991 and
grad-clip 1.0 → 0.979 both rescue it. So the recurrent depth arm is trainable across the whole ladder
at 3e-3; that is this module's default.

**Arms** (all share the S5/Z120 word data + held-out split from `depth_composition_toy`):
- `shared_readout`  — `RecurrentComposer`: shared block + ONE readout for every length. The literal
  "one readout serving every capacity level" that FAILED for width (`verdict_variable_width_mse.md`
  §3) — the transfer-prediction FAILURE candidate.
- `per_length_head` — `RecurrentPerLengthHead`: the SAME shared block + a separate readout head per
  length (the #2 pattern). The charter's per-input-depth net.
- `wide_shallow`    — a fixed 1-hidden-layer MLP on the word one-hot ZERO-PADDED to the max length
  (no per-input depth). The width-substitution control; should stall on long words.

**Pre-registered bars (from `depth.md` D1b, fixed before this ran):**
- G1 *deep fits:* `per_length_head` holds held-out acc ≥ `FIT_ACC` (0.90) across the WHOLE ladder.
- G2 *width stalls:* `wide_shallow` falls ≤ `STALL_ACC` (0.60) at the long-word end.
- G3 *graded dial:* the (`per_length_head` − `wide_shallow`) acc GAP increases monotonically with
  length (the separation widens as the per-input depth requirement grows — the dial itself).
- G4 *control shows NO divergence:* on Z120, BOTH the recurrent and wide-shallow arms stay ≥ `FIT_ACC`
  at every length (gap ≈ 0) — width substitutes for the solvable group, so the S5 divergence in G3 is
  specifically about non-solvability/depth, not "long sequences are hard." (Run `--group z120`.)
- Transfer prediction CONFIRMED iff `shared_readout` fails G1 where `per_length_head` passes.

Metric: held-out (unseen-word) accuracy per length; chance = 1/|G| = 1/120. Convergence-gated on the
across-lengths mean val cross-entropy (`convergence.py`, full-trajectory rule, best weights restored).

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/depth_graded_toy.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/depth_graded_toy.py \
        --pilot --group s5 --seed 0            # + --group z120 for the G4 control
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import sys
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root

import convergence as cvg  # noqa: E402 — shared full-trajectory convergence gate
from depth_composition_toy import (  # noqa: E402 — reuse the composition toy's validated building blocks
    DEFAULT_TRAIN_FRAC,
    REC_STATE_WIDTH,
    WIDE_WIDTH,
    Group,
    RecurrentComposer,
    build_wide_shallow_clf,
    count_params,
    make_word_data,
)

LR_DEFAULT = 3e-3  # canonical LR (n=10 spike: 1e-2 stalls the deep unroll, 3e-3 reaches 0.99)
MAX_EPOCHS = 40000
CHECK_EVERY = 250
PATIENCE = 10
MIN_DELTA = 1e-4
LENGTH_LADDER = (4, 6, 8, 10)  # word-length ladder = the per-input depth dial (ℓ=2 has too few words)
FIT_ACC = 0.90  # G1/G4 "fits"
STALL_ACC = 0.60  # G2 "stalls"
N_CLASSES = 120  # |S5| = |Z120| = 120 group elements (the classification output width)


class Arm(enum.StrEnum):
    """Which architecture arm a row reports (closed set)."""

    SHARED_READOUT = "shared_readout"  # RecurrentComposer: shared block + one readout for all lengths
    PER_LENGTH_HEAD = "per_length_head"  # shared block + per-length readout heads (#2 pattern)
    WIDE_SHALLOW = "wide_shallow"  # fixed MLP on zero-padded word (width-substitution control)


class RecurrentPerLengthHead(nn.Module):
    """Shared recurrent block (identical to `RecurrentComposer`'s) + a separate readout head per length.

    The depth edition of `SharedTrunkPerWidthHeadNet`: the block folds the word left-to-right exactly
    as `RecurrentComposer` does (shared across every length), but `forward(x, length)` reads out through
    `heads[str(length)]`. Isolates readout interference — the block is shared, only the head varies.
    """

    def __init__(self, width: int, n_gen: int, n_classes: int, lengths: tuple[int, ...], block_hidden: int | None = None) -> None:
        """Shared 2-layer transition block over `[state(width), onehot_gen(n_gen)] -> width`, one readout head per length."""
        super().__init__()
        self.width = int(width)
        self.n_gen = int(n_gen)
        h = block_hidden if block_hidden is not None else width
        self.block = nn.Sequential(nn.Linear(width + n_gen, h), nn.Tanh(), nn.Linear(h, width))
        self.heads = nn.ModuleDict({str(int(ell)): nn.Linear(width, n_classes) for ell in lengths})

    def forward(self, x_flat: torch.Tensor, length: int) -> torch.Tensor:
        """Fold `x_flat` `(N, ℓ*n_gen)` with the shared block, then read out through the length-`length` head."""
        n = x_flat.shape[0]
        seq = x_flat.view(n, -1, self.n_gen)
        state = torch.zeros(n, self.width, device=x_flat.device)
        for t in range(seq.shape[1]):
            state = torch.tanh(self.block(torch.cat([state, seq[:, t, :]], dim=1)))
        return self.heads[str(int(length))](state)


def _pad_onehot(x_flat: np.ndarray, seq_len: int, n_gen: int, max_len: int) -> np.ndarray:
    """Zero-pad a flattened one-hot word `(n, seq_len*n_gen)` to `(n, max_len*n_gen)` (pad positions all-zero)."""
    n = x_flat.shape[0]
    a = x_flat.reshape(n, seq_len, n_gen)
    out = np.zeros((n, max_len, n_gen), dtype=np.float32)
    out[:, :seq_len, :] = a
    return out.reshape(n, max_len * n_gen)


def make_graded_data(group: Group, lengths: tuple[int, ...], seed: int, train_frac: float = DEFAULT_TRAIN_FRAC) -> dict[int, dict]:
    """Per-length S5/Z120 word datasets (each via `make_word_data`, independent words via a per-length seed)."""
    return {int(ell): make_word_data(group, int(ell), seed * 100 + int(ell), train_frac) for ell in lengths}


def _train_mixed(
    net: nn.Module, tensors: dict[int, tuple], fwd: Callable[[torch.Tensor, int], torch.Tensor], lr: float, device: str, max_epochs: int = MAX_EPOCHS,
) -> tuple[cvg.ConvergenceResult, dict[int, dict]]:
    """Full-batch Adam + summed-over-lengths CE, convergence-gated on the mean-over-lengths val CE.

    `tensors[ℓ] = (x_tr, y_tr, x_val, y_val)`; `fwd(x, ℓ)` produces logits from `net`. Returns the
    convergence result and `{ℓ: {train_acc, val_acc}}` (top-1 on held-out and train words per length).
    """
    net.to(device)
    lengths = list(tensors.keys())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    def step_fn() -> None:
        opt.zero_grad()
        loss = sum(ce(fwd(tensors[ell][0], ell), tensors[ell][1]) for ell in lengths)
        loss.backward()
        opt.step()

    def val_fn() -> float:
        net.eval()
        with torch.no_grad():
            v = sum(ce(fwd(tensors[ell][2], ell), tensors[ell][3]).item() for ell in lengths) / len(lengths)
        net.train()
        return v

    result = cvg.fit_to_convergence(net, step_fn, val_fn, max_epochs=max_epochs, check_every=CHECK_EVERY, patience=PATIENCE, min_delta=MIN_DELTA)
    net.eval()
    accs: dict[int, dict] = {}
    with torch.no_grad():
        for ell in lengths:
            x_tr, y_tr, x_val, y_val = tensors[ell]
            accs[ell] = {
                "train_acc": float((fwd(x_tr, ell).argmax(1) == y_tr).float().mean().item()),
                "val_acc": float((fwd(x_val, ell).argmax(1) == y_val).float().mean().item()),
            }
    return result, accs


def _to_tensors(per_len: dict[int, dict], device: str, pad_to: int | None = None) -> dict[int, tuple]:
    """Materialize `{ℓ: (x_tr, y_tr, x_val, y_val)}` float/long tensors; if `pad_to`, zero-pad the one-hot to that length."""
    out: dict[int, tuple] = {}
    for ell, d in per_len.items():
        x_tr, x_val = d["x_tr"], d["x_val"]
        if pad_to is not None:
            x_tr = _pad_onehot(x_tr, d["seq_len"], d["n_gen"], pad_to)
            x_val = _pad_onehot(x_val, d["seq_len"], d["n_gen"], pad_to)
        out[ell] = (
            torch.as_tensor(x_tr, dtype=torch.float32, device=device),
            torch.as_tensor(d["y_tr"], dtype=torch.long, device=device),
            torch.as_tensor(x_val, dtype=torch.float32, device=device),
            torch.as_tensor(d["y_val"], dtype=torch.long, device=device),
        )
    return out


def run_pilot(group: Group, lengths: tuple[int, ...], seed: int, lr: float = LR_DEFAULT, device: str = "cpu") -> dict:
    """Train the 3 arms on mixed-length `group` words; return per-arm per-length held-out accuracy + bar checks."""
    per_len = make_graded_data(group, lengths, seed)
    n_classes = per_len[lengths[0]]["n_classes"]
    n_gen = per_len[lengths[0]]["n_gen"]
    max_len = max(lengths)

    # per-length tensors (recurrent arms) and zero-padded tensors (wide-shallow arm).
    rec_tensors = _to_tensors(per_len, device)
    pad_tensors = _to_tensors(per_len, device, pad_to=max_len)

    arms: dict[str, dict] = {}

    torch.manual_seed(1000 * seed + 1)
    shared = RecurrentComposer(REC_STATE_WIDTH, n_gen, n_classes)
    r_sh, acc_sh = _train_mixed(shared, rec_tensors, lambda x, _ell: shared(x), lr, device)
    arms[Arm.SHARED_READOUT.value] = {"params": count_params(shared), "trustworthy": bool(r_sh.trustworthy), "convergence": r_sh.summary(), "by_length": acc_sh}

    torch.manual_seed(1000 * seed + 2)
    perlen = RecurrentPerLengthHead(REC_STATE_WIDTH, n_gen, n_classes, lengths)
    r_pl, acc_pl = _train_mixed(perlen, rec_tensors, perlen, lr, device)
    arms[Arm.PER_LENGTH_HEAD.value] = {"params": count_params(perlen), "trustworthy": bool(r_pl.trustworthy), "convergence": r_pl.summary(), "by_length": acc_pl}

    torch.manual_seed(1000 * seed + 3)
    wide = build_wide_shallow_clf(WIDE_WIDTH, max_len * n_gen, n_classes)
    r_w, acc_w = _train_mixed(wide, pad_tensors, lambda x, _ell: wide(x), lr, device)
    arms[Arm.WIDE_SHALLOW.value] = {"params": count_params(wide), "trustworthy": bool(r_w.trustworthy), "convergence": r_w.summary(), "by_length": acc_w}

    bars = _check_bars(arms, lengths)
    return {
        "group": group.value, "seed": seed, "lr": lr, "lengths": list(lengths), "max_len": max_len,
        "n_classes": n_classes, "chance": 1.0 / n_classes,
        "state_width": REC_STATE_WIDTH, "wide_width": WIDE_WIDTH,
        "fit_acc": FIT_ACC, "stall_acc": STALL_ACC, "arms": arms, "bars": bars,
    }


def _check_bars(arms: dict[str, dict], lengths: tuple[int, ...]) -> dict:
    """Evaluate G1-G4 + the transfer prediction from the per-arm per-length val accuracies (within one group)."""
    def val(arm: str, ell: int) -> float:
        return arms[arm]["by_length"][ell]["val_acc"]

    long_end = max(lengths)
    pl_all = [val(Arm.PER_LENGTH_HEAD.value, ell) for ell in lengths]
    sh_all = [val(Arm.SHARED_READOUT.value, ell) for ell in lengths]
    gaps = [val(Arm.PER_LENGTH_HEAD.value, ell) - val(Arm.WIDE_SHALLOW.value, ell) for ell in lengths]
    g3_monotonic = all(gaps[i + 1] >= gaps[i] - 1e-6 for i in range(len(gaps) - 1))

    return {
        "G1_deep_fits": bool(min(pl_all) >= FIT_ACC),
        "G2_width_stalls": bool(val(Arm.WIDE_SHALLOW.value, long_end) <= STALL_ACC),
        "G3_graded_gap_monotonic": bool(g3_monotonic),
        "G4_control_no_divergence": bool(min(pl_all) >= FIT_ACC and min(val(Arm.WIDE_SHALLOW.value, ell) for ell in lengths) >= FIT_ACC),
        "transfer_shared_readout_fails": bool(min(sh_all) < FIT_ACC and min(pl_all) >= FIT_ACC),
        "gaps_by_length": {int(ell): round(g, 4) for ell, g in zip(lengths, gaps, strict=True)},
    }


DEFAULT_OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "D_TOY_PROBES")


def run_and_save_pilot(group: Group, lengths: tuple[int, ...], seed: int, lr: float, out_dir: str, device: str = "cpu") -> dict:
    """`run_pilot` + immediate JSON land (standing clause: land results the moment they exist)."""
    os.makedirs(out_dir, exist_ok=True)
    pilot = run_pilot(group, lengths, seed, lr=lr, device=device)
    path = os.path.join(out_dir, f"depth_graded_pilot_{group.value}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(pilot, f, indent=2)
    return pilot, path


def run_wide_only_pilot(group: Group, lengths: tuple[int, ...], seed: int, wide_width: int, lr: float = LR_DEFAULT, device: str = "cpu") -> dict:
    """Train ONLY the `wide_shallow` arm at `wide_width` (param-matched control run; no bars, no recurrent arms)."""
    per_len = make_graded_data(group, lengths, seed)
    n_classes = per_len[lengths[0]]["n_classes"]
    n_gen = per_len[lengths[0]]["n_gen"]
    max_len = max(lengths)

    pad_tensors = _to_tensors(per_len, device, pad_to=max_len)

    torch.manual_seed(1000 * seed + 3)
    wide = build_wide_shallow_clf(wide_width, max_len * n_gen, n_classes)
    r_w, acc_w = _train_mixed(wide, pad_tensors, lambda x, _ell: wide(x), lr, device)
    arm_row = {"params": count_params(wide), "trustworthy": bool(r_w.trustworthy), "convergence": r_w.summary(), "by_length": acc_w}

    return {
        "group": group.value, "seed": seed, "lr": lr, "lengths": list(lengths), "max_len": max_len,
        "n_classes": n_classes, "chance": 1.0 / n_classes,
        "wide_width": wide_width,
        "arms": {Arm.WIDE_SHALLOW.value: arm_row},
    }


def run_and_save_wide_only_pilot(group: Group, lengths: tuple[int, ...], seed: int, lr: float, wide_width: int, out_dir: str, device: str = "cpu") -> tuple[dict, str]:
    """`run_wide_only_pilot` + immediate JSON land to a collision-free `_w{wide_width}` filename."""
    os.makedirs(out_dir, exist_ok=True)
    pilot = run_wide_only_pilot(group, lengths, seed, wide_width, lr=lr, device=device)
    path = os.path.join(out_dir, f"depth_graded_pilot_{group.value}_seed{seed}_w{wide_width}.json")
    with open(path, "w") as f:
        json.dump(pilot, f, indent=2)
    return pilot, path


def run_selftest() -> bool:
    """Net wiring + data-pipeline shape checks (no training): per-length-head + padding + graded data."""
    ok = True
    lengths = (4, 6)
    # per-length-head net returns (N, n_classes) for each length, using the right head.
    net = RecurrentPerLengthHead(8, 4, N_CLASSES, lengths)
    for ell in lengths:
        out = net(torch.zeros(3, ell * 4), ell)
        shape_ok = tuple(out.shape) == (3, N_CLASSES)
        print(f"[graded selftest] per_length_head length={ell} out {tuple(out.shape)} (expect (3,120))  {'PASS' if shape_ok else 'FAIL'}")
        ok = shape_ok and ok
    # padding: a length-4 word padded to 6 keeps its first 4 positions and zeros the rest.
    x = np.eye(4, dtype=np.float32)[np.array([[0, 1, 2, 3]])].reshape(1, 16)
    padded = _pad_onehot(x, 4, 4, 6)
    pad_ok = padded.shape == (1, 24) and np.allclose(padded.reshape(1, 6, 4)[:, :4], x.reshape(1, 4, 4)) and np.allclose(padded.reshape(1, 6, 4)[:, 4:], 0.0)
    print(f"[graded selftest] pad 4->6 shape+content  {'PASS' if pad_ok else 'FAIL'}")
    ok = pad_ok and ok
    # graded data: distinct per-length datasets with disjoint train/val and the shared class count.
    data = make_graded_data(Group.S5, lengths, seed=0)
    shapes_ok = all(data[ell]["x_tr"].shape[1] == ell * 4 and data[ell]["x_tr"].shape[0] > 0 and data[ell]["x_val"].shape[0] > 0 for ell in lengths)
    classes_ok = all(d["n_classes"] == N_CLASSES for d in data.values())
    data_ok = shapes_ok and classes_ok
    print(f"[graded selftest] graded data per-length shapes/split  {'PASS' if data_ok else 'FAIL'}")
    ok = data_ok and ok
    print(f"[graded selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Parse args and run `--selftest` or a `--pilot`, else print help."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="No-training net-wiring / padding / data-shape checks.")
    parser.add_argument("--pilot", action="store_true", help="Train the 3 arms on mixed-length words; report per-length held-out acc + bars.")
    parser.add_argument("--group", type=str, choices=[g.value for g in Group], default=Group.S5.value, help="Group (default s5; z120 = G4 control).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT, help="Adam learning rate (default 3e-3, the canonical fix).")
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTH_LADDER), help="Word-length ladder (default 4 6 8 10).")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Directory for pilot JSON output.")
    parser.add_argument("--wide-width", type=int, default=WIDE_WIDTH, help=f"wide_shallow arm width (default {WIDE_WIDTH}); non-default trains ONLY that arm.")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    if args.pilot:
        device = os.environ.get("AUTOML_DEVICE", "cpu")
        group = Group(args.group)
        lengths = tuple(args.lengths)

        if args.wide_width != WIDE_WIDTH:
            pilot, path = run_and_save_wide_only_pilot(group, lengths, args.seed, args.lr, args.wide_width, args.out_dir, device=device)
            print(f"[graded] wrote {path}")
            print(f"  group={pilot['group']} seed={pilot['seed']} lr={pilot['lr']} lengths={pilot['lengths']} chance={pilot['chance']:.4f} wide_width={pilot['wide_width']}")
            row = pilot["arms"][Arm.WIDE_SHALLOW.value]
            cells = "  ".join(f"L{ell}={row['by_length'][ell]['val_acc']:.3f}" for ell in lengths)
            print(f"  {Arm.WIDE_SHALLOW.value:>16} (prm {row['params']:>6}, trust={row['trustworthy']}): {cells}")
            sys.exit(0)

        pilot, path = run_and_save_pilot(group, lengths, args.seed, args.lr, args.out_dir, device=device)
        print(f"[graded] wrote {path}")
        print(f"  group={pilot['group']} seed={pilot['seed']} lr={pilot['lr']} lengths={pilot['lengths']} chance={pilot['chance']:.4f}")
        for arm, row in pilot["arms"].items():
            cells = "  ".join(f"L{ell}={row['by_length'][ell]['val_acc']:.3f}" for ell in lengths)
            print(f"  {arm:>16} (prm {row['params']:>6}, trust={row['trustworthy']}): {cells}")
        print(f"  bars: {pilot['bars']}")
        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()
