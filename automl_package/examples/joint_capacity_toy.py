"""Joint width+depth toy (J0 Step 2) — ONE net serving a per-input 2-D capacity dial (state-width w, unroll-depth T).

Design doc: `docs/joint_capacity/joint_toy_design.md` (§2 the joint net; §3 the three candidates -- this
module builds J-1 and J-2, J-3 is deferred; §4 the pilot bars S1/W-bar/D-bar/X-bar; §5 the J-1 arithmetic
verification; §6 the G-JOINT gate). Plan: `docs/plans/capacity_programme/width-depth.md` Task J0.
Mirrors `depth_selection_toy.py`'s structure/idioms exactly, extended for the width dial + the (w,T) grid.

**The joint net (design §2).** A weight-shared recurrent block folds the per-step input; the 2-D dial is
(state-width w, unroll-depth T):
```
state_0 = 0  (REC_STATE_WIDTH-wide)
for t in 1..T:  state = tanh(block([mask_w(state)?, input_t]))   # unroll T = DEPTH dial
logits_track_k = per_width_head_w(mask_w(state))[:, k, :]         # read at width w = WIDTH dial
```
Depth dial T = unroll count, exactly `depth_composition_toy.RecurrentComposer`'s mechanism
(`depth_composition_toy.py:295`): `state = tanh(block([state, input_t]))`, weight-shared across steps.
Width dial w = active state units, exactly `nested_width_net.SharedTrunkPerWidthHeadNet`'s mechanism
(`nested_width_net.py:222`): prefix-mask `state[:, w:] = 0`, one PER-WIDTH head per rung
(`Linear(REC_STATE_WIDTH, K_MAX*n_classes)`). `mask_w` placement is the ONLY thing distinguishing J-1
from J-2 (`WidthMode`, design §3's "hedge on the key risk"):
  - READOUT (J-1, PRIMARY): the block folds at full state_width every step; `mask_w` is applied only
    once, right before the per-width head reads the state. The fold itself never depends on w, so ONE
    fold pass produces every width rung's state (`JointCapacityNet.forward_grid` caches it).
  - BLOCK (J-2, HEDGE on J-1's risk that a narrow readout free-rides on a full-width computation):
    `mask_w` is applied to the state BEFORE it re-enters the block at every single fold step, so a
    narrow width genuinely cannot carry more than w units of information forward. This requires one
    separate fold per width rung (the fold trajectory itself now depends on w).

**Data (design §3 J-1 + `docs/joint_capacity/j1_arithmetic_check.py`).** K_MAX=4 parallel track slots.
Per input: draw width-demand A ~ U{1..K_MAX} and depth-demand T* ~ U(T_STAR_LADDER) INDEPENDENTLY. A
active slots each get a length-L A5-involution word with realized commitment == T* (`depth_selection_toy
.sample_stratum`, reused verbatim); the K_MAX-A inactive slots are all-NOOP (5th symbol = identity).
Per-step input = concat over the K_MAX slots of onehot_5(letter) -> K_MAX*5 = 20 dims/step, 200 dims over
L=10 steps. Label = tuple of K_MAX track PRODUCTS (inactive tracks -> identity).

**Performance (mandatory per the build brief -- do NOT call `sample_stratum` per active track per
input, which rebuilds the taboo DP tables every call and takes hours at pilot scale).** `_presample_pools`
draws ONE pool of `n_pool` words per T* stratum (3 calls total, `sample_stratum` + `word_all_prefix_
products`, both called exactly once per stratum); `assemble_cell` then draws WITH REPLACEMENT from the
pre-sampled pool for every (A, T*) cell -- O(1) index lookups, no DP rebuilds. `j1_arithmetic_check.
build_input`'s per-input `sample_stratum` calls are NOT reused here for this reason (that module's own
one-off n=4000 arithmetic check can afford it; a real pilot at thousands of inputs per cell cannot).

**Training (mirrors `depth_selection_toy.train_anytime`'s root-cause fix).** Each anytime exit T in
`T_LADDER` is trained against the T-step RUNNING (prefix) product PER TRACK -- never the impossible
full-word target for T < t*(x) -- for BOTH active tracks (real running product) and inactive tracks
(constant identity, trivially achievable at every T, "still supervised" per the build brief). Loss =
mean cross-entropy over every (width rung, T) cell of the training grid, gradient-clipped
(`depth_selection_toy.GRAD_CLIP_MAX_NORM`, the L=10 GD-trainable-wall fix), convergence-gated
(`convergence.fit_to_convergence`) on the same grid's mean val CE. The DEPLOY answer is always the
word's FULL product (`prefix_products[..., -1]`, since T_LADDER's last rung is L), matching depth_
selection_toy's deploy-vs-train-target split.

**Bars (design §4; thresholds FROZEN in `joint_frozen_bar_spec.md`).** S1 (substrate fit: per-track
active accuracy per (A,T*) cell). W-bar (width dial is REAL, not a free-ride): the acc(w_max)-acc(w_min)
gap on all-active-correct is measured on the A==W_BAR_SUBPOP subpopulation ONLY (bimodal there -- ~0 real
vs ~acc(w_max) free-ride; A=1 would only dilute it) AND Spearman(deployed-w, A) over all A. D-bar (depth
dial is graded): reuses `depth_selection_toy.s2_ceiling_a5`'s exact knee/ceiling formula PER TRACK at
w=w_max, pooled per T* stratum -- single-track, so the 0.35 ceiling is calibrated verbatim (a per-input
all-active metric would need an A-dependent 0.25^A ceiling) -- plus Spearman(deployed-T, T*). X-bar (a
distilled 2-D router beats the val-selected best fixed (w,T) AND strictly beats both marginal routers on
mean-w*T compute at accuracy within 2*SE). Router labels/deploy machinery reuse `sinc_width_experiment._cheapest_within_
tolerance_labels` and `depth_selection_toy._VectorRouterMLP`/`_train_vector_router`/`_route_vector`
verbatim (metric-agnostic: fed classification error = 1-accuracy). The two MARGINAL routers (width-only
at fixed T=L; depth-only at fixed w=w_max) reuse `sinc_width_experiment._deploy_bar_mse` verbatim (their
column keys are literal, collision-free w or T values). The JOINT 2-D router does NOT reuse `_deploy_bar
_mse` verbatim: (w,T) pairs collide on the w*T compute proxy (e.g. w=16,T=4 and w=32,T=2 both cost 64),
so its error table must stay keyed by the (w,T) TUPLE (never lossy) -- `_deploy_bar_joint` below reuses
`sinc_width_experiment._plain_boot_se` (the actual reusable numeric primitive) for the paired-bootstrap
SE and reimplements only the tuple-vs-float key handling `_deploy_bar_mse` cannot do unmodified.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/joint_capacity_toy.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/joint_capacity_toy.py --probe arithmetic
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/joint_capacity_toy.py \
        --probe pilot --width-mode readout --n-per-cell 1000 --max-epochs 40000   # the real pilot (main thread only)
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
_REPO_ROOT = os.path.dirname(os.path.dirname(_EXAMPLES_DIR))
sys.path.insert(0, _REPO_ROOT)  # repo root, so `import automl_package` works
_DOCS_JOINT_DIR = os.path.join(_REPO_ROOT, "docs", "joint_capacity")
sys.path.insert(0, _DOCS_JOINT_DIR)  # so `import j1_arithmetic_check` works (design §5 arithmetic, not under examples/)

import convergence as cvg  # noqa: E402 — full-trajectory convergence gate, shared with every capacity toy
import depth_selection_toy as dst  # noqa: E402 — reuse the certified A5 machinery + anytime-net conventions (D8b), no reinvention
import j1_arithmetic_check as j1  # noqa: E402 — reuse the certified J-1 data arithmetic (design §5) verbatim
import sinc_width_experiment as sw  # noqa: E402 — reuse `_cheapest_within_tolerance_labels`/`_deploy_bar_mse`/`_plain_boot_se` (metric-agnostic)
from depth_composition_toy import REC_STATE_WIDTH  # noqa: E402 — the certified recurrent-block state width (depth_composition_toy.py:101)

# ---------------------------------------------------------------------------
# Construction constants -- reused from `j1_arithmetic_check`/`depth_selection_toy` wherever the SAME
# quantity already exists there (K_MAX/T_STAR_LADDER/L/NOOP from J-1's own arithmetic module; N_CLASSES/
# T_LADDER/convergence hyperparameters/LR/GRAD_CLIP from the certified D8b anytime-net convention).
# ---------------------------------------------------------------------------

K_MAX = j1.K_MAX  # width dial: number of parallel track slots (4)
T_STAR_LADDER = j1.T_STAR_LADDER  # depth-demand generative strata (6, 8, 10)
L = j1.L  # word length (10) -- the A5 GD-trainable wall (MOD-1), shared with the depth-selection toy
NOOP = j1.NOOP  # padding symbol for inactive tracks (identity; the 4 involutions are indices 0..3)
N_CLASSES = dst._A5_N_CLASSES  # 60
N_GEN_REAL = dst._A5_N_GEN  # 4 real A5-involution generators per track
N_SYMBOLS = N_GEN_REAL + 1  # + NOOP -- each track's per-step onehot alphabet size (5)
T_LADDER = dst.T_LADDER  # anytime unroll-depth exits: (2, 4, 6, 8, 10) -- T=L=10 is the deploy/full rung

WIDTH_LADDER = (16, 32, 48, 64)  # width dial ladder -- module constant so the orchestrator can edit it pre-pilot without touching logic
if WIDTH_LADDER[-1] != REC_STATE_WIDTH:
    raise ValueError("WIDTH_LADDER's top rung must equal the recurrent state width")

TRAIN_FRAC = dst.TRAIN_FRAC  # 50/50 train/val split per cell (certified-toy convention)
LR_DEFAULT = dst.LR_DEFAULT  # canonical recurrent-arm LR (depth_graded_toy's L=10 trainability fix)
MAX_EPOCHS_DEFAULT = dst.MAX_EPOCHS_DEFAULT
GRAD_CLIP_MAX_NORM = dst.GRAD_CLIP_MAX_NORM
DELTA_TIE = sw.DELTA_TIE  # cheapest-within-tolerance router tolerance, reused verbatim

N_PER_CELL_DEFAULT = 1000  # design §4's "~1000/cell" pilot-cost estimate
N_POOL_PER_STRATUM_MIN_DEFAULT = 2000  # floor pool size so even a tiny smoke run has some word diversity

S1_FIT_ACC = 0.90  # S1 substrate bar: full (w_max, T=L) held-out ACTIVE-track acc >= this, every (A,T*) cell
W_BAR_GAP = 0.30  # FROZEN (joint_frozen_bar_spec §2): acc(w_max) - acc(w_min) on the A==W_BAR_SUBPOP subset >= this, at fixed full T=L
W_BAR_SUBPOP = 4  # FROZEN: W-bar gap measured ONLY on the A==4 held-out subpop -- the free-ride is bimodal/diagnostic only at maximal A (A=1 cannot discriminate)
WIDTH_SPEARMAN_MIN = 0.7  # W-bar: rank corr between the router's deployed w and the realized width-demand A
DEPTH_SPEARMAN_MIN = 0.7  # D-bar: rank corr between the router's deployed T and the realized depth-demand T*

DEFAULT_OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "J_TOY_PROBES")


class WidthMode(enum.StrEnum):
    """Where `mask_w` is applied (design §3) -- the ONE difference between J-1 and J-2."""

    READOUT = "readout"  # J-1 (PRIMARY): block folds at full state_width; mask applied only at the per-width head
    BLOCK = "block"  # J-2 (HEDGE): mask applied to the state at EVERY fold step -- a genuinely narrower computation


# ---------------------------------------------------------------------------
# Data — pre-sampled pools per T* stratum (performance, see module docstring), then per-(A,T*)-cell
# assembly by drawing WITH REPLACEMENT from the pool (no `sample_stratum` call per input).
# ---------------------------------------------------------------------------


def _presample_pools(t_star_ladder: tuple[int, ...], n_pool: int, seed: int, grp: dict | None = None) -> dict[int, dict]:
    """One `sample_stratum` + one `word_all_prefix_products` call PER stratum (never per input)."""
    grp = grp or dst._A5_GRP
    pools = {}
    for t in t_star_ladder:
        drawn = dst.sample_stratum(t, n_pool, seed=seed * 1000 + t, grp=grp)
        prefix_products = dst.word_all_prefix_products(drawn["word_gen_ids"], grp)  # (n_pool, L)
        pools[t] = {"word_gen_ids": drawn["word_gen_ids"], "labels": drawn["labels"], "prefix_products": prefix_products}
    return pools


def _random_active_slots(n: int, k_max: int, a_active: int, rng: np.random.Generator) -> np.ndarray:
    """`(n, a_active)` sorted, no-duplicate slot indices in `[0, k_max)` -- one random subset per row, fully vectorized."""
    order = np.argsort(rng.random((n, k_max)), axis=1)
    chosen = order[:, :a_active]
    chosen.sort(axis=1)
    return chosen


def assemble_cell(a_active: int, t_star: int, n: int, pool: dict, rng: np.random.Generator, k_max: int = K_MAX, word_len: int = L, grp: dict | None = None) -> dict:
    """Assembles `n` J-1 inputs for cell (a_active, t_star), drawing active-track words from the pre-sampled `pool`.

    Inactive (K_MAX - a_active) slots are filled with NOOP and the identity element at every position --
    "still supervised" (module docstring): trivially achievable at every T, per the build brief.
    """
    grp = grp or dst._A5_GRP
    n_pool = pool["word_gen_ids"].shape[0]
    tracks = np.full((n, k_max, word_len), NOOP, dtype=np.int64)
    active_mask = np.zeros((n, k_max), dtype=bool)
    prefix_products = np.full((n, k_max, word_len), grp["identity"], dtype=np.int64)

    chosen_slots = _random_active_slots(n, k_max, a_active, rng)  # (n, a_active)
    pool_idx = rng.integers(0, n_pool, size=(n, a_active))
    rows = np.arange(n)
    for j in range(a_active):
        slot = chosen_slots[:, j]
        idx = pool_idx[:, j]
        tracks[rows, slot, :] = pool["word_gen_ids"][idx]
        prefix_products[rows, slot, :] = pool["prefix_products"][idx]
        active_mask[rows, slot] = True

    return {
        "tracks": tracks, "active_mask": active_mask, "prefix_products": prefix_products,
        "a_active": np.full(n, a_active, dtype=np.int64), "t_star": np.full(n, t_star, dtype=np.int64),
    }


def _onehot_tracks(tracks: np.ndarray, n_symbols: int = N_SYMBOLS) -> np.ndarray:
    """`(n, K_MAX, L)` track symbol ids -> `(n, L*K_MAX*n_symbols)` flattened per-step onehot (design §3: concat over slots, then over steps)."""
    n, k_max, l_len = tracks.shape
    onehot = np.eye(n_symbols, dtype=np.float32)[tracks]  # (n, k_max, l_len, n_symbols)
    onehot = onehot.transpose(0, 2, 1, 3)  # (n, l_len, k_max, n_symbols)
    return onehot.reshape(n, l_len * k_max * n_symbols)


def build_joint_dataset(
    t_star_ladder: tuple[int, ...] = T_STAR_LADDER, k_max: int = K_MAX, n_per_cell: int = N_PER_CELL_DEFAULT,
    seed: int = 0, n_pool_per_stratum: int | None = None, train_frac: float = TRAIN_FRAC, grp: dict | None = None,
) -> dict:
    """Builds the full `k_max * len(t_star_ladder)`-cell (A, T*) dataset, each cell train/val split.

    `n_pool_per_stratum` defaults to `max(n_per_cell * k_max, N_POOL_PER_STRATUM_MIN_DEFAULT)` -- enough
    draws-with-replacement diversity for any `n_per_cell` (from a --selftest-scale smoke run to a real
    pilot) without a separate required CLI flag.
    """
    grp = grp or dst._A5_GRP
    n_pool = n_pool_per_stratum or max(n_per_cell * k_max, N_POOL_PER_STRATUM_MIN_DEFAULT)
    pools = _presample_pools(t_star_ladder, n_pool, seed, grp)
    rng = np.random.default_rng(seed)
    per_cell = {}
    for a in range(1, k_max + 1):
        for t in t_star_ladder:
            cell = assemble_cell(a, t, n_per_cell, pools[t], rng, k_max=k_max, grp=grp)
            n_tr = round(train_frac * n_per_cell)
            x = _onehot_tracks(cell["tracks"])
            per_cell[(a, t)] = {
                "x_tr": x[:n_tr], "x_val": x[n_tr:],
                "active_mask_tr": cell["active_mask"][:n_tr], "active_mask_val": cell["active_mask"][n_tr:],
                "prefix_products_tr": cell["prefix_products"][:n_tr], "prefix_products_val": cell["prefix_products"][n_tr:],
                "a_active_tr": cell["a_active"][:n_tr], "a_active_val": cell["a_active"][n_tr:],
                "t_star_tr": cell["t_star"][:n_tr], "t_star_val": cell["t_star"][n_tr:],
            }
    return {"per_cell": per_cell, "k_max": k_max, "t_star_ladder": list(t_star_ladder), "n_gen_per_step": k_max * N_SYMBOLS, "n_classes": len(grp["elements"])}


def _pool_cells(data: dict, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenates `x/active_mask/prefix_products/a_active/t_star` across every (A,T*) cell for `split` ('tr' or 'val')."""
    xs, ams, pps, a_s, t_s = [], [], [], [], []
    for cell in data["per_cell"].values():
        xs.append(cell[f"x_{split}"])
        ams.append(cell[f"active_mask_{split}"])
        pps.append(cell[f"prefix_products_{split}"])
        a_s.append(cell[f"a_active_{split}"])
        t_s.append(cell[f"t_star_{split}"])
    return np.concatenate(xs), np.concatenate(ams), np.concatenate(pps), np.concatenate(a_s), np.concatenate(t_s)


# ---------------------------------------------------------------------------
# The joint net (design §2) — weight-shared recurrent block (state width REC_STATE_WIDTH) + one
# per-width head per WIDTH_LADDER rung. `forward_grid` computes every (width rung, T) cell requested;
# in READOUT mode the fold is width-independent, so it is run ONCE and reused across every width rung.
# ---------------------------------------------------------------------------


class JointCapacityNet(nn.Module):
    """Shared recurrent block + per-width heads, with `mask_w` placed per `width_mode` (design §2/§3)."""

    def __init__(
        self, width_mode: WidthMode, n_gen_per_step: int, n_classes: int, k_max: int = K_MAX,
        width_ladder: tuple[int, ...] = WIDTH_LADDER, state_width: int = REC_STATE_WIDTH, block_hidden: int | None = None,
    ) -> None:
        """Builds the shared 2-layer transition block (`RecurrentComposer`'s block shape) plus one `Linear(state_width, k_max*n_classes)` head per width rung."""
        super().__init__()
        self.width_mode = WidthMode(width_mode)
        self.n_gen = int(n_gen_per_step)
        self.n_classes = int(n_classes)
        self.k_max = int(k_max)
        self.state_width = int(state_width)
        self.width_ladder = tuple(width_ladder)
        h = block_hidden if block_hidden is not None else self.state_width
        self.block = nn.Sequential(nn.Linear(self.state_width + self.n_gen, h), nn.Tanh(), nn.Linear(h, self.state_width))
        self.heads = nn.ModuleList(nn.Linear(self.state_width, self.k_max * self.n_classes) for _ in self.width_ladder)

    def _mask(self, state: torch.Tensor, w: int) -> torch.Tensor:
        """Prefix-masks `state[:, w:] = 0` (`nested_width_net.SharedTrunkPerWidthHeadNet`'s mechanism); no-ops at the full width rung."""
        if w >= self.state_width:
            return state
        masked = state.clone()
        masked[:, w:] = 0.0
        return masked

    def _fold_states(self, x_flat: torch.Tensor, t_ladder: tuple[int, ...], w_for_block: int | None) -> dict[int, torch.Tensor]:
        """Folds the shared block over `x_flat`, snapshotting the state at every `t` in `t_ladder`.

        `w_for_block=None` (READOUT): the state is never masked mid-fold, so this single pass is valid
        for every width rung. `w_for_block=w` (BLOCK): the state is masked to width `w` before feeding
        it back into the block at EVERY step, so a caller needing multiple widths must call this once
        per width.
        """
        n = x_flat.shape[0]
        seq = x_flat.view(n, -1, self.n_gen)
        max_t = max(t_ladder)
        state = torch.zeros(n, self.state_width, device=x_flat.device)
        states_by_t = {}
        for t in range(1, max_t + 1):
            block_in = self._mask(state, w_for_block) if w_for_block is not None else state
            state = torch.tanh(self.block(torch.cat([block_in, seq[:, t - 1, :]], dim=1)))
            if t in t_ladder:
                states_by_t[t] = state
        return states_by_t

    def head_logits(self, state: torch.Tensor, width_rung_idx: int) -> torch.Tensor:
        """`(N, k_max, n_classes)` logits at width rung `width_rung_idx`, reading the readout-masked state."""
        w = self.width_ladder[width_rung_idx]
        masked = self._mask(state, w)
        return self.heads[width_rung_idx](masked).view(-1, self.k_max, self.n_classes)

    def forward_grid(self, x_flat: torch.Tensor, t_ladder: tuple[int, ...] = T_LADDER, width_ladder_idx: tuple[int, ...] | None = None) -> dict[tuple[int, int], torch.Tensor]:
        """`{(width_rung_idx, t): (N, k_max, n_classes) logits}` for every requested width rung x `t_ladder` cell."""
        idxs = width_ladder_idx if width_ladder_idx is not None else tuple(range(len(self.width_ladder)))
        out: dict[tuple[int, int], torch.Tensor] = {}
        if self.width_mode is WidthMode.READOUT:
            states_by_t = self._fold_states(x_flat, t_ladder, w_for_block=None)
            for w_idx in idxs:
                for t in t_ladder:
                    out[(w_idx, t)] = self.head_logits(states_by_t[t], w_idx)
        else:
            for w_idx in idxs:
                states_by_t = self._fold_states(x_flat, t_ladder, w_for_block=self.width_ladder[w_idx])
                for t in t_ladder:
                    out[(w_idx, t)] = self.head_logits(states_by_t[t], w_idx)
        return out


# ---------------------------------------------------------------------------
# Training (mirrors `depth_selection_toy.train_anytime`'s root-cause fix, extended over the width grid).
# ---------------------------------------------------------------------------


def train_joint_net(
    data: dict, seed: int, device: str, width_mode: WidthMode, lr: float = LR_DEFAULT, max_epochs: int = MAX_EPOCHS_DEFAULT,
    t_ladder: tuple[int, ...] = T_LADDER, width_ladder: tuple[int, ...] = WIDTH_LADDER,
) -> tuple[JointCapacityNet, cvg.ConvergenceResult]:
    """Trains the joint net on the pooled (all-cell) data; loss = mean CE over every (width rung, T) cell of the grid, per track."""
    x_tr, _am_tr, pp_tr, _a_tr, _t_tr = _pool_cells(data, "tr")
    x_val, _am_val, pp_val, _a_val, _t_val = _pool_cells(data, "val")

    torch.manual_seed(seed)
    net = JointCapacityNet(width_mode, data["n_gen_per_step"], data["n_classes"], k_max=data["k_max"], width_ladder=width_ladder)
    net.to(device)

    x_tr_t = torch.as_tensor(x_tr, dtype=torch.float32, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_tr_by_t = {t: torch.as_tensor(pp_tr[:, :, t - 1], dtype=torch.long, device=device) for t in t_ladder}
    y_val_by_t = {t: torch.as_tensor(pp_val[:, :, t - 1], dtype=torch.long, device=device) for t in t_ladder}

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    n_terms = len(width_ladder) * len(t_ladder)

    def _grid_loss(grid: dict[tuple[int, int], torch.Tensor], y_by_t: dict[int, torch.Tensor]) -> torch.Tensor:
        terms = [ce(grid[(w_idx, t)].reshape(-1, net.n_classes), y_by_t[t].reshape(-1)) for w_idx in range(len(width_ladder)) for t in t_ladder]
        return sum(terms) / n_terms

    def step_fn() -> None:
        opt.zero_grad()
        loss = _grid_loss(net.forward_grid(x_tr_t, t_ladder=t_ladder), y_tr_by_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        opt.step()

    def val_fn() -> float:
        net.eval()
        with torch.no_grad():
            v = float(_grid_loss(net.forward_grid(x_val_t, t_ladder=t_ladder), y_val_by_t).item())
        net.train()
        return v

    result = cvg.fit_to_convergence(net, step_fn, val_fn, max_epochs=max_epochs, check_every=dst.CHECK_EVERY, patience=dst.PATIENCE, min_delta=dst.MIN_DELTA)
    return net, result


# ---------------------------------------------------------------------------
# Per-input accuracy grid — one forward pass per width-mode-appropriate fold (see `forward_grid`), then
# per-track and all-active-tracks-correct booleans, keyed by (width rung, T), for every bar below.
# ---------------------------------------------------------------------------


def per_input_accuracy_grid(
    net: JointCapacityNet, x: np.ndarray, prefix_products: np.ndarray, active_mask: np.ndarray, device: str,
    t_ladder: tuple[int, ...] = T_LADDER, width_ladder_idx: tuple[int, ...] | None = None,
) -> dict:
    """`{track_correct, all_active_correct}`, each keyed `(width_rung_idx, t)`, plus the full (deploy-target) label.

    Correctness always compares against the word's FULL product (`prefix_products[..., -1]`, T=L's
    training target), never the T-step training target for T < L -- the root-cause distinction
    `depth_selection_toy.per_input_accuracy_table` also makes.
    """
    net.eval()
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    full_label = prefix_products[:, :, -1]  # (N, K_MAX) full product per track (T=L is the last t_ladder rung)
    full_label_t = torch.as_tensor(full_label, dtype=torch.long, device=device)
    track_correct: dict[tuple[int, int], np.ndarray] = {}
    all_active_correct: dict[tuple[int, int], np.ndarray] = {}
    with torch.no_grad():
        grid = net.forward_grid(x_t, t_ladder=t_ladder, width_ladder_idx=width_ladder_idx)
        for key, logits in grid.items():
            pred = logits.argmax(-1)
            correct = (pred == full_label_t).cpu().numpy()  # (N, K_MAX)
            track_correct[key] = correct
            all_active_correct[key] = np.where(active_mask, correct, True).all(axis=1)
    return {"track_correct": track_correct, "all_active_correct": all_active_correct, "full_label": full_label}


# ---------------------------------------------------------------------------
# Bars (design §4).
# ---------------------------------------------------------------------------


def check_s1_bar(
    track_correct: dict, a_active_val: np.ndarray, t_star_val: np.ndarray, active_mask: np.ndarray, w_max_idx: int, full_t: int = L, fit_acc: float = S1_FIT_ACC,
) -> dict:
    """S1 -- at full (w_max, T=L), held-out ACTIVE-track accuracy >= `fit_acc` on every (A, T*) cell."""
    correct = track_correct[(w_max_idx, full_t)]  # (N, K_MAX) bool
    per_cell = {}
    for a in range(1, K_MAX + 1):
        for t in T_STAR_LADDER:
            mask = (a_active_val == a) & (t_star_val == t)
            if not mask.any():
                continue
            active_entries = correct[mask][active_mask[mask]]
            per_cell[f"a={a},t={t}"] = float(active_entries.mean()) if active_entries.size else float("nan")
    s1_pass = bool(per_cell) and all(v >= fit_acc for v in per_cell.values())
    return {"per_cell": per_cell, "s1_pass": s1_pass}


def check_w_bar(
    all_active_correct: dict, a_active_val: np.ndarray, deployed_w: np.ndarray, w_min_idx: int, w_max_idx: int,
    full_t: int = L, gap: float = W_BAR_GAP, spearman_min: float = WIDTH_SPEARMAN_MIN, subpop: int = W_BAR_SUBPOP,
) -> dict:
    """W-bar (make-or-break) -- the width dial is REAL, not a narrow-readout free-ride.

    The accuracy gap `acc(w_max) - acc(w_min)` is measured ONLY on the A==`subpop` held-out subpopulation
    (frozen spec §2): at maximal A the all-active-correct metric is bimodal -- ~0 if a narrow read cannot
    hold all A tracks (real dial), ~acc(w_max) if it free-rides -- so the gap cleanly separates the two.
    A=1 inputs are trivially all-correct at every width and would only dilute the gap. The companion
    Spearman(deployed-w, A) recovery check is over the FULL held-out set (all A levels).
    """
    a_sub = a_active_val == subpop
    acc_w_max = float(all_active_correct[(w_max_idx, full_t)][a_sub].mean()) if a_sub.any() else float("nan")
    acc_w_min = float(all_active_correct[(w_min_idx, full_t)][a_sub].mean()) if a_sub.any() else float("nan")
    gap_val = acc_w_max - acc_w_min
    gap_pass = bool(np.isfinite(gap_val) and gap_val >= gap)
    rho = float(spearmanr(deployed_w, a_active_val.astype(np.float64)).statistic)
    rho_pass = bool(np.isfinite(rho) and rho >= spearman_min)
    w_bar_pass = bool(gap_pass and rho_pass)
    return {
        "subpop_a": subpop, "n_subpop": int(a_sub.sum()), "acc_w_max": acc_w_max, "acc_w_min": acc_w_min,
        "gap": gap_val, "gap_pass": gap_pass, "width_spearman": rho, "width_spearman_pass": rho_pass, "w_bar_pass": w_bar_pass,
    }


def check_d_bar(
    track_correct: dict, active_mask: np.ndarray, t_star_val: np.ndarray, deployed_t: np.ndarray, w_max_idx: int, ceiling: float,
    full_t: int = L, knee_fraction: float = dst.S2_KNEE_FRACTION, spearman_min: float = DEPTH_SPEARMAN_MIN,
) -> dict:
    """D-bar (frozen spec §4 FLAG-2) -- PER-TRACK depth knee at full width w_max, pooled per T* stratum.

    Measured on ACTIVE tracks only, at w=w_max (so an insufficient-width read never confounds the depth
    knee), pooled per T* stratum across all A. Because this is single-track accuracy, the reused
    `ceiling`=Bayes(g=2)+0.10=0.35 is calibrated verbatim -- a per-input all-active metric would instead
    need an A-dependent 0.25^A ceiling. Knee: acc(T=T*) >= knee_fraction * acc(T=L); ceiling:
    acc(T=T*-2) <= ceiling; plus Spearman(deployed-T, T*).
    """
    def _per_track_acc(t_read: int, stratum_mask: np.ndarray) -> float:
        corr = track_correct[(w_max_idx, t_read)][stratum_mask]  # (n, K_MAX) bool
        entries = corr[active_mask[stratum_mask]]  # active tracks only
        return float(entries.mean()) if entries.size else float("nan")

    per_stratum = {}
    for t in T_STAR_LADDER:
        mask = t_star_val == t
        if not mask.any():
            continue
        acc_at_t = _per_track_acc(t, mask)
        acc_at_full = _per_track_acc(full_t, mask)
        knee_ratio = acc_at_t / acc_at_full if acc_at_full > 0 else float("nan")
        t_minus_2 = t - dst.BAYES_G_FOR_CEILING
        acc_at_t_minus_2 = _per_track_acc(t_minus_2, mask) if t_minus_2 in T_LADDER else None
        per_stratum[f"t={t}"] = {
            "acc_at_t_star": acc_at_t, "acc_at_full_t": acc_at_full, "knee_ratio": knee_ratio, "knee_pass": bool(np.isfinite(knee_ratio) and knee_ratio >= knee_fraction),
            "acc_at_t_star_minus_2": acc_at_t_minus_2, "ceiling": ceiling, "ceiling_pass": bool(acc_at_t_minus_2 is None or acc_at_t_minus_2 <= ceiling),
        }
    knee_ceiling_pass = bool(per_stratum) and all(v["knee_pass"] and v["ceiling_pass"] for v in per_stratum.values())
    rho = float(spearmanr(deployed_t, t_star_val.astype(np.float64)).statistic)
    rho_pass = bool(np.isfinite(rho) and rho >= spearman_min)
    d_bar_pass = bool(knee_ceiling_pass and rho_pass)
    return {
        "per_stratum": per_stratum, "knee_ceiling_pass": knee_ceiling_pass,
        "depth_spearman": rho, "depth_spearman_pass": rho_pass, "d_bar_pass": d_bar_pass,
    }


# ---------------------------------------------------------------------------
# Router / deploy (design §3.5-style S4 no-oracle distillation, X-bar). Reuses `depth_selection_toy._
# VectorRouterMLP`/`_train_vector_router`/`_route_vector` (a raw-input vector router, already generalized
# there beyond the scalar-input routers the width toys use) and `sinc_width_experiment._cheapest_within_
# tolerance_labels` verbatim. The two MARGINAL routers additionally reuse `sinc_width_experiment._deploy_
# bar_mse` verbatim (their columns are keyed by a literal, collision-free w or T value); the JOINT 2-D
# router needs `_deploy_bar_joint` below (module docstring explains why `_deploy_bar_mse` can't be reused
# unmodified for a 2-D grid with compute-proxy collisions).
# ---------------------------------------------------------------------------

JOINT_COLUMNS: list[tuple[int, int]] = sorted(
    ((w_idx, t) for w_idx in range(len(WIDTH_LADDER)) for t in T_LADDER),
    key=lambda wt: (WIDTH_LADDER[wt[0]] * wt[1], WIDTH_LADDER[wt[0]], wt[1]),
)  # every (width rung, T) column, ascending by compute w*T (ties broken by w then T) -- the router's label order


def fit_width_only_router(err_tr_by_w: dict[int, np.ndarray], x_tr: np.ndarray, seed: int, device: str, width_ladder: tuple[int, ...] = WIDTH_LADDER) -> dst._VectorRouterMLP:
    """Marginal router: routes width w only, at fixed full T=L."""
    err_matrix = np.stack([err_tr_by_w[w] for w in width_ladder], axis=1)
    labels = sw._cheapest_within_tolerance_labels(err_matrix, DELTA_TIE)
    return dst._train_vector_router(x_tr, len(width_ladder), device, labels, seed)


def fit_depth_only_router(err_tr_by_t: dict[int, np.ndarray], x_tr: np.ndarray, seed: int, device: str, t_ladder: tuple[int, ...] = T_LADDER) -> dst._VectorRouterMLP:
    """Marginal router: routes depth T only, at fixed w=w_max."""
    t_sorted = tuple(sorted(t_ladder))
    err_matrix = np.stack([err_tr_by_t[t] for t in t_sorted], axis=1)
    labels = sw._cheapest_within_tolerance_labels(err_matrix, DELTA_TIE)
    return dst._train_vector_router(x_tr, len(t_sorted), device, labels, seed)


def fit_joint_router(
    err_tr_by_col: dict[tuple[int, int], np.ndarray], x_tr: np.ndarray, seed: int, device: str, columns: list[tuple[int, int]] = JOINT_COLUMNS,
) -> dst._VectorRouterMLP:
    """Primary router: routes the full (width rung, T) grid."""
    err_matrix = np.stack([err_tr_by_col[c] for c in columns], axis=1)
    labels = sw._cheapest_within_tolerance_labels(err_matrix, DELTA_TIE)
    return dst._train_vector_router(x_tr, len(columns), device, labels, seed)


def _deploy_bar_joint(err_hardpick: np.ndarray, executed_compute: np.ndarray, err_by_col: dict[tuple[int, int], np.ndarray], width_ladder: tuple[int, ...] = WIDTH_LADDER) -> dict:
    """Joint-router twin of `sw._deploy_bar_mse`, keyed by the (w,T) TUPLE (never lossy under w*T collisions).

    Reuses `sw._plain_boot_se` for the paired-bootstrap SE (the actual reusable numeric primitive);
    the tuple-vs-float key handling is the one piece `_deploy_bar_mse` cannot do unmodified (see module
    docstring).
    """
    per_col_mse = {c: float(v.mean()) for c, v in err_by_col.items()}
    best_fixed_col = min(per_col_mse, key=per_col_mse.get)
    best_fixed_compute = width_ladder[best_fixed_col[0]] * best_fixed_col[1]
    mse_best_fixed = per_col_mse[best_fixed_col]
    mse_hardpick = float(err_hardpick.mean())
    mean_executed_compute = float(executed_compute.mean())

    paired_delta = err_hardpick - err_by_col[best_fixed_col]
    se_paired = sw._plain_boot_se(paired_delta)
    accuracy_preserved = bool(mse_hardpick <= mse_best_fixed + 2.0 * se_paired)
    compute_saved = bool(mean_executed_compute < best_fixed_compute)
    return {
        "mse_hardpick": mse_hardpick, "mean_executed_compute": mean_executed_compute, "mse_best_fixed": mse_best_fixed,
        "best_fixed_w": width_ladder[best_fixed_col[0]], "best_fixed_t": best_fixed_col[1], "best_fixed_compute": best_fixed_compute,
        "se_paired": se_paired, "accuracy_preserved": accuracy_preserved, "compute_saved": compute_saved, "deploy_pass": bool(accuracy_preserved and compute_saved),
    }


def check_x_bar(
    joint_deploy: dict, err_hardpick_joint: np.ndarray, err_hardpick_width: np.ndarray, err_hardpick_depth: np.ndarray,
    width_only_mean_compute: float, depth_only_mean_compute: float,
) -> dict:
    """X-bar (frozen spec §5): joint router beats the val-selected best-fixed (w,T) AND strictly beats BOTH marginals.

    Marginal-beat = joint mean w*T compute STRICTLY < the marginal's AND joint accuracy no worse within
    2*SE_paired (frozen spec: accuracy within 2 SE, strictly less compute). width-only always pays T=L,
    depth-only always pays w=w_max, so a genuine 2-D dial saves compute against both. The best-fixed leg
    is `joint_deploy["deploy_pass"]` (`_deploy_bar_joint`, val-selected best-fixed column).
    """
    joint_compute = float(joint_deploy["mean_executed_compute"])
    joint_err_mean = float(err_hardpick_joint.mean())

    def _beats(marginal_err: np.ndarray, marginal_compute: float) -> dict:
        se = sw._plain_boot_se(err_hardpick_joint - marginal_err)  # paired: >0 where joint is worse
        acc_no_worse = bool(joint_err_mean <= float(marginal_err.mean()) + 2.0 * se)
        compute_strictly_less = bool(joint_compute < marginal_compute)
        return {
            "acc_no_worse_2se": acc_no_worse, "compute_strictly_less": compute_strictly_less, "se_paired": se,
            "marginal_mean_compute": marginal_compute, "beats": bool(acc_no_worse and compute_strictly_less),
        }

    vs_width = _beats(err_hardpick_width, width_only_mean_compute)
    vs_depth = _beats(err_hardpick_depth, depth_only_mean_compute)
    return {
        "joint_deploy_pass": bool(joint_deploy["deploy_pass"]), "joint_mean_compute": joint_compute,
        "vs_width_only": vs_width, "vs_depth_only": vs_depth,
        "beats_width_only": vs_width["beats"], "beats_depth_only": vs_depth["beats"],
        "x_bar_pass": bool(joint_deploy["deploy_pass"] and vs_width["beats"] and vs_depth["beats"]),
    }


# ---------------------------------------------------------------------------
# Pilot orchestration — builds the dataset, trains, computes the full accuracy grid, distills all three
# routers, evaluates all four bars, writes the JSON (design §4's pilot output).
# ---------------------------------------------------------------------------


def run_pilot(width_mode: WidthMode, n_per_cell: int, max_epochs: int, lr: float, seed: int, device: str, out_dir: str, n_pool_per_stratum: int | None = None) -> dict:
    """Runs the full J-toy pilot: train, grid-eval, distill (joint + both marginal routers), evaluate S1/W-bar/D-bar/X-bar, write JSON."""
    data = build_joint_dataset(n_per_cell=n_per_cell, seed=seed, n_pool_per_stratum=n_pool_per_stratum)
    net, result = train_joint_net(data, seed, device, width_mode, lr=lr, max_epochs=max_epochs)

    x_tr, am_tr, pp_tr, _a_tr, _t_tr = _pool_cells(data, "tr")  # _a_tr/_t_tr unused: router labels are trained on error only, no oracle
    x_val, am_val, pp_val, a_val, t_val = _pool_cells(data, "val")

    grid_tr = per_input_accuracy_grid(net, x_tr, pp_tr, am_tr, device)
    grid_val = per_input_accuracy_grid(net, x_val, pp_val, am_val, device)

    w_max_idx = len(WIDTH_LADDER) - 1
    w_min_idx = 0

    # --- width-only marginal router (fixed full T=L) ---
    err_tr_by_w = {WIDTH_LADDER[i]: (~grid_tr["all_active_correct"][(i, L)]).astype(np.float64) for i in range(len(WIDTH_LADDER))}
    err_val_by_w = {WIDTH_LADDER[i]: (~grid_val["all_active_correct"][(i, L)]).astype(np.float64) for i in range(len(WIDTH_LADDER))}
    width_router = fit_width_only_router(err_tr_by_w, x_tr, seed, device)
    col_idx_w = dst._route_vector(width_router, x_val, device)
    executed_w = np.array([WIDTH_LADDER[c] for c in col_idx_w], dtype=np.float64)
    err_matrix_val_w = np.stack([err_val_by_w[w] for w in WIDTH_LADDER], axis=1)
    err_hardpick_w = err_matrix_val_w[np.arange(len(col_idx_w)), col_idx_w]
    width_only_deploy = sw._deploy_bar_mse(err_hardpick_w, executed_w, err_val_by_w)

    # --- depth-only marginal router (fixed w=w_max) ---
    t_sorted = tuple(sorted(T_LADDER))
    err_tr_by_t = {t: (~grid_tr["all_active_correct"][(w_max_idx, t)]).astype(np.float64) for t in t_sorted}
    err_val_by_t = {t: (~grid_val["all_active_correct"][(w_max_idx, t)]).astype(np.float64) for t in t_sorted}
    depth_router = fit_depth_only_router(err_tr_by_t, x_tr, seed, device)
    col_idx_t = dst._route_vector(depth_router, x_val, device)
    executed_t = np.array([t_sorted[c] for c in col_idx_t], dtype=np.float64)
    err_matrix_val_t = np.stack([err_val_by_t[t] for t in t_sorted], axis=1)
    err_hardpick_t = err_matrix_val_t[np.arange(len(col_idx_t)), col_idx_t]
    depth_only_deploy = sw._deploy_bar_mse(err_hardpick_t, executed_t, err_val_by_t)

    # --- joint 2-D router ---
    err_tr_by_col = {c: (~grid_tr["all_active_correct"][c]).astype(np.float64) for c in JOINT_COLUMNS}
    err_val_by_col = {c: (~grid_val["all_active_correct"][c]).astype(np.float64) for c in JOINT_COLUMNS}
    joint_router = fit_joint_router(err_tr_by_col, x_tr, seed, device)
    col_idx_j = dst._route_vector(joint_router, x_val, device)
    deployed_w = np.array([WIDTH_LADDER[JOINT_COLUMNS[c][0]] for c in col_idx_j], dtype=np.float64)
    deployed_t = np.array([JOINT_COLUMNS[c][1] for c in col_idx_j], dtype=np.float64)
    executed_compute = deployed_w * deployed_t
    err_matrix_val_j = np.stack([err_val_by_col[c] for c in JOINT_COLUMNS], axis=1)
    err_hardpick_j = err_matrix_val_j[np.arange(len(col_idx_j)), col_idx_j]
    joint_deploy = _deploy_bar_joint(err_hardpick_j, executed_compute, err_val_by_col)

    ceiling = dst.s2_ceiling_a5(dst.compute_arithmetic_table())
    s1 = check_s1_bar(grid_val["track_correct"], a_val, t_val, am_val, w_max_idx)
    w_bar = check_w_bar(grid_val["all_active_correct"], a_val, deployed_w, w_min_idx, w_max_idx)
    d_bar = check_d_bar(grid_val["track_correct"], am_val, t_val, deployed_t, w_max_idx, ceiling)
    width_only_mean_compute = float((executed_w * L).mean())  # width-only router pays full T=L
    depth_only_mean_compute = float((WIDTH_LADDER[-1] * executed_t).mean())  # depth-only router pays full w=w_max
    x_bar = check_x_bar(joint_deploy, err_hardpick_j, err_hardpick_w, err_hardpick_t, width_only_mean_compute, depth_only_mean_compute)

    router_features = {
        "input": "flattened raw one-hot (K_MAX tracks x N_SYMBOLS) per step, L steps",
        "in_dim": int(x_val.shape[1]),
        "n_joint_columns": len(JOINT_COLUMNS),
        "label_source": "cheapest-within-tolerance over the held-out per-(w,T) error table (sw._cheapest_within_tolerance_labels); NO (A,T*) oracle",
        "oracle_labels_used": False,
    }
    pilot_pass = bool(s1["s1_pass"] and w_bar["w_bar_pass"] and d_bar["d_bar_pass"] and x_bar["x_bar_pass"])

    out = {
        "seed": seed, "width_mode": width_mode.value, "n_per_cell": n_per_cell, "convergence": result.summary(),
        "s1": s1, "w_bar": w_bar, "d_bar": d_bar, "x_bar": x_bar,
        "width_only_deploy": width_only_deploy, "depth_only_deploy": depth_only_deploy, "joint_deploy": joint_deploy,
        "router_features": router_features, "pilot_pass": pilot_pass,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"joint_{width_mode.value}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return {"path": path, **out}


# ---------------------------------------------------------------------------
# Selftest — reuses `depth_selection_toy.run_selftest` verbatim (A5 axioms, sampler assertion, prefix
# products, single-track anytime-net shapes), plus joint-specific fast checks.
# ---------------------------------------------------------------------------


def _check_prefix_products_consistency() -> bool:
    """The pre-sampled pool's cached prefix products must match a from-scratch `dst.word_all_prefix_products` recompute."""
    pools = _presample_pools((6,), n_pool=50, seed=1)
    pool = pools[6]
    recomputed = dst.word_all_prefix_products(pool["word_gen_ids"])
    ok = bool(np.array_equal(recomputed, pool["prefix_products"]))
    print(f"[joint selftest] pool prefix-products match dst.word_all_prefix_products: {ok}  {'PASS' if ok else 'FAIL'}")
    return ok


def _check_inactive_track_identity() -> bool:
    """Inactive (NOOP-filled) track slots must carry the identity element at every position -- "still supervised" per the build brief."""
    a_active_test, t_star_test, n_test = 2, 8, 20
    pools = _presample_pools(T_STAR_LADDER, n_pool=200, seed=0)
    rng = np.random.default_rng(0)
    cell = assemble_cell(a_active_test, t_star_test, n_test, pools[t_star_test], rng)
    inactive = ~cell["active_mask"]
    ident_ok = bool((cell["prefix_products"][inactive] == dst._A5_IDENTITY).all())
    noop_ok = bool((cell["tracks"][inactive] == NOOP).all())
    active_count_ok = bool((cell["active_mask"].sum(axis=1) == a_active_test).all())
    ok = ident_ok and noop_ok and active_count_ok
    print(f"[joint selftest] inactive-track identity: ident_ok={ident_ok} noop_fill_ok={noop_ok} active_count_ok={active_count_ok}  {'PASS' if ok else 'FAIL'}")
    return ok


def _check_joint_net_forward_shapes() -> bool:
    """Joint net forward-shape check at two (w,T) combos, in BOTH width modes (small dummy net, no training)."""
    ok = True
    n_gen_per_step, n_classes, k_max = 8, 6, 2
    width_ladder = (2, 4)
    t_probe = (2, 4)
    l_test = max(t_probe)
    for mode in WidthMode:
        net = JointCapacityNet(mode, n_gen_per_step, n_classes, k_max=k_max, width_ladder=width_ladder, state_width=4)
        x = torch.zeros(3, l_test * n_gen_per_step)
        grid = net.forward_grid(x, t_ladder=t_probe)
        for w_idx in range(len(width_ladder)):
            for t in t_probe:
                out = grid[(w_idx, t)]
                shape_ok = tuple(out.shape) == (3, k_max, n_classes)
                expect = (3, k_max, n_classes)
                print(f"[joint selftest] net forward mode={mode.value} w_idx={w_idx} T={t}: out {tuple(out.shape)} (expect {expect})  {'PASS' if shape_ok else 'FAIL'}")
                ok = ok and shape_ok
    return ok


def run_selftest() -> bool:
    """All fast (seconds), no-real-training checks: the reused D8b suite + joint-specific data/net checks."""
    ok = True
    ok = dst.run_selftest() and ok
    ok = _check_prefix_products_consistency() and ok
    ok = _check_inactive_track_identity() and ok
    ok = _check_joint_net_forward_shapes() and ok
    print(f"[joint selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parses args and runs `--selftest` or one `--probe {arithmetic,pilot}`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Fast (seconds), no-real-training checks.")
    parser.add_argument("--probe", type=str, choices=["arithmetic", "pilot"], default=None, help="Which probe to run.")
    parser.add_argument("--width-mode", type=str, choices=[m.value for m in WidthMode], default=WidthMode.READOUT.value, help="J-1 (readout) or J-2 (block) mask placement.")
    parser.add_argument("--n-per-cell", type=int, default=N_PER_CELL_DEFAULT, help="Inputs per (A,T*) cell (12 cells total).")
    parser.add_argument("--n-pool-per-stratum", type=int, default=None, help="Words pre-sampled per T* stratum pool (default: auto-scaled from --n-per-cell).")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS_DEFAULT, help="Max epochs for the joint-net convergence gate.")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Directory for probe JSON output.")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    if args.probe == "arithmetic":
        j1.main()  # reuse the certified prints verbatim (design §5 numbers)
        out = {
            "orthogonality": j1.check_orthogonality(), "class_balance": j1.class_balance(),
            "joint_bayes_ceiling": j1.joint_bayes_ceiling(), "width_info_floor": j1.width_info_floor(),
        }
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "joint_arithmetic_j1.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[joint] wrote {path}")
        sys.exit(0)

    if args.probe == "pilot":
        device = os.environ.get("AUTOML_DEVICE", "cpu")
        width_mode = WidthMode(args.width_mode)
        result = run_pilot(width_mode, args.n_per_cell, args.max_epochs, args.lr, args.seed, device, args.out_dir, n_pool_per_stratum=args.n_pool_per_stratum)
        print(f"[joint pilot] width_mode={width_mode.value} seed={args.seed} trustworthy={result['convergence']['trustworthy']} diverged={result['convergence']['diverged']}")
        print(f"  S1 pass={result['s1']['s1_pass']}  W-bar pass={result['w_bar']['w_bar_pass']}")
        print(f"  D-bar pass={result['d_bar']['d_bar_pass']}  X-bar pass={result['x_bar']['x_bar_pass']}")
        print(f"  pilot_pass={result['pilot_pass']}")
        print(f"[joint] wrote {result['path']}")
        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()
