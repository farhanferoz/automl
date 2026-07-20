r"""F6 — FlexNN-vs-MoE comparison battery (rescopes M3; UNGATED from G-JOINT).

`docs/plans/capacity_programme/flexnn-core.md` Task F6. Compares four contenders on three toys,
each contender matched to a common reference capacity in TWO regimes (total params, executed
FLOPs), 3 seeds minimum, convergence-gated (MASTER Decision 9):

  * **Toys**: (1) the certified width toy (`nested_width_net.make_hetero`, config from
    `width-cert.md`'s W9 canonical headline: `shared_trunk`/MSE, `n_train=1500`, `n_test=500`,
    `sigma=0.05`, `w_max=12`); (2) `nested_width_net.make_hetero3` (WP-3 noisy-easy negative
    control, `n_train=2250`, `n_test=750`); (3) the A5 depth-composition toy in its CE form
    (`depth_composition_toy.make_word_data(Group.A5, ...)`) — classification over 60 group
    elements, the toy that needed `moe_regression.py`'s new `task=ce` flag.
  * **Contenders**: `FlexibleHiddenLayersNN` (routed, post-F1/F3), `FlexibleWidthNN` (routed),
    MoE top-2 primary (with a top-1 ablation, same `expert_hidden` — Switch's "deliberate
    simplification" of Mixtral's top-2, NOT independently re-matched), and a val-selected
    best-fixed-capacity baseline read off the toy's OWN "natural" dial net (no extra training).
  * **Matching**: each toy's "natural" dial net (`FlexibleWidthNN` for the two regression toys,
    `FlexibleHiddenLayersNN` for the classification toy) at its LARGEST configured capacity is
    the reference `capacity_accounting` (S2) target; MoE's `expert_hidden` is solved once per
    regime via `moe_regression.match_to_reference`'s `h_from_params_only`/`h_from_flops_only`
    closed-form intermediates (params-matched and FLOPs-matched are genuinely different builds
    only when those two solves diverge — both are reported either way).
  * **Tuned-alpha rerun clause** (carried verbatim from the old M5 spec): before this driver's own
    JSON asserts a "MoE underperforms" reading for a toy/seed cell, it automatically reruns that
    cell's best MoE variant at two alternative aux-loss alpha values
    (`TUNED_ALPHA_CANDIDATES`) and records whether tuning closes the gap — never a hand-wavy
    claim from the frozen alpha=1e-2 run alone.
  * **G-JOINT is DROPPED from this grid** (open research problem, decided separately in
    `flexnn-core.md` MASTER Decision 5) — only the two CERTIFIED 1-D dials (width, depth) are in
    scope; no joint (width, depth) toy is run here.

Package-model training bypasses `BaseModel.fit()`'s Optuna/early-stopping wrapper entirely and
calls `_fit_single(x_train, y_train, x_val=..., y_val=...)` directly, then replays the returned
per-epoch trajectory through `automl_package.utils.convergence.ConvergenceTracker` for the
canonical converged/trustworthy verdict — the established capacity-programme pattern
(`automl_package/examples/flexnn_revalidation.py`), not a new convention. MoE training reuses
`convergence.fit_to_convergence` directly (the same gate `depth_composition_toy.py`'s `train_clf`
uses), full-batch Adam, closing over `moe_regression.training_loss`.

Held-out generalization split per toy (adapting each toy's OWN established convention to the
package-model harness's need for an explicit validation set):
  * Width toy / hetero3: an INDEPENDENT test draw at `seed+500` (exactly `width-cert.md`'s own
    convention) plus a local 80/20 shuffle-split of the `n_train` pool into (train, val) for
    early stopping / `fit_router` / best-fixed selection — that val split is new (the raw
    research harness in `kdropout_converged_width_experiment.py` does not need one), stated here
    explicitly rather than silently reusing the same 1500/2250 points for both training and
    model selection.
  * A5 depth toy: `make_word_data`'s own held-out (unseen-word) split IS final test (never used
    for any selection decision); the TRAIN half is further locally split 80/20 into (train, val)
    for early stopping / `fit_router` / best-fixed selection.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/moe_flexnn_comparison.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/moe_flexnn_comparison.py \
        --toys width,hetero3,a5_depth --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import capacity_accounting as ca  # noqa: E402
import convergence as cvg  # noqa: E402 — full-trajectory convergence gate (automl_package.utils.convergence re-export)
import depth_composition_toy as dct  # noqa: E402
import moe_regression as moe  # noqa: E402
import nested_width_net as nwn  # noqa: E402

from automl_package.enums import ActivationFunction, DepthRegularization, LayerSelectionMethod, TaskType, UncertaintyMethod, WidthSelectionMethod  # noqa: E402
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN  # noqa: E402
from automl_package.models.flexible_width_network import FlexibleWidthNN  # noqa: E402

OUT_DIR = os.path.join(_EXAMPLES_DIR, "moe_comparison_results")

SEEDS_DEFAULT = (0, 1, 2)  # S3: 3 seeds minimum
VAL_FRAC = 0.2  # local train-pool -> (train, val) split fraction, both regression toys and A5's train half

# --- MoE frozen config (M1, moe_regression.py) reused verbatim here ---------------------------
N_EXPERTS = moe.N_EXPERTS_PRIMARY
TOP_K_PRIMARY = moe.TOP_K_PRIMARY
TOP_K_ABLATION = moe.TOP_K_ABLATION
ALPHA_FROZEN = moe.ALPHA
TUNED_ALPHA_CANDIDATES = (1e-3, 1e-1)  # tuned-alpha rerun clause: bracket the frozen 1e-2 by 10x either side
MOE_FAIL_MARGIN = 0.05  # relative margin: MoE's best variant this much worse than the best non-MoE contender triggers the rerun clause


class ToyKey(enum.StrEnum):
    """Closed set of this battery's three toys (G-JOINT dropped — see module docstring)."""

    WIDTH = "width"  # certified width toy (width-cert.md ledger), regression
    HETERO3 = "hetero3"  # WP-3 noisy-easy negative control, regression
    A5_DEPTH = "a5_depth"  # A5 group word-composition, classification (CE)


class MatchRegime(enum.StrEnum):
    """Closed set: which S2 quantity `expert_hidden` is solved to match."""

    PARAMS = "params_matched"
    FLOPS = "flops_matched"


class ContenderKind(enum.StrEnum):
    """Closed set of result-row labels this driver emits per toy/seed."""

    FLEX_HIDDEN_ROUTED = "flexible_hidden_layers_nn_routed"
    FLEX_WIDTH_ROUTED = "flexible_width_nn_routed"
    MOE_TOP2 = "moe_top2"
    MOE_TOP1_ABLATION = "moe_top1_ablation"
    BEST_FIXED = "best_fixed_val_selected"


# ---------------------------------------------------------------------------
# Per-toy configuration — data builders + architecture/training hyperparameters. Toys 1/2 share
# the same 1-D regression architecture choices; toy 3 (classification) has its own.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToyConfig:
    """Everything `run_toy_seed` needs for one toy: data shape, task, architectures, training budget."""

    key: ToyKey
    task_type: TaskType
    d_in: int
    output_size: int  # 1 for regression, n_classes for classification
    reference_family: str  # "width" or "depth" — which FlexNN family is this toy's natural dial (MoE matching target)
    width_ladder: tuple[int, ...]
    depth_max: int
    hidden_size: int
    learning_rate: float
    n_epochs_cap: int
    patience: int
    min_delta: float
    batch_size: int
    moe_max_epochs: int
    moe_check_every: int
    moe_patience: int
    moe_min_delta: float
    moe_lr: float


A5_SEQ_LEN = 5  # module default (depth_composition_toy.py --seq-len default) — 4**5=1024 words, GD-learnable range
A5_N_GEN = len(dct.A5_GENERATORS)  # 4 generators (involution alphabet, depth_composition_toy.py)
A5_D_IN = A5_SEQ_LEN * A5_N_GEN  # flattened one-hot word width
A5_N_CLASSES = len(dct.build_group(dct.Group.A5)["elements"])  # |A5| = 5!/2 = 60, derived not hardcoded

TOY_CONFIGS: dict[ToyKey, ToyConfig] = {
    ToyKey.WIDTH: ToyConfig(
        key=ToyKey.WIDTH, task_type=TaskType.REGRESSION, d_in=1, output_size=1, reference_family="width",
        width_ladder=(2, 4, 8, 12), depth_max=4, hidden_size=16, learning_rate=0.01,
        n_epochs_cap=1500, patience=60, min_delta=1e-4, batch_size=64,
        moe_max_epochs=6000, moe_check_every=100, moe_patience=8, moe_min_delta=1e-4, moe_lr=0.01,
    ),
    ToyKey.HETERO3: ToyConfig(
        key=ToyKey.HETERO3, task_type=TaskType.REGRESSION, d_in=1, output_size=1, reference_family="width",
        width_ladder=(2, 4, 8, 12), depth_max=4, hidden_size=16, learning_rate=0.01,
        n_epochs_cap=1500, patience=60, min_delta=1e-4, batch_size=64,
        moe_max_epochs=6000, moe_check_every=100, moe_patience=8, moe_min_delta=1e-4, moe_lr=0.01,
    ),
    ToyKey.A5_DEPTH: ToyConfig(
        key=ToyKey.A5_DEPTH, task_type=TaskType.CLASSIFICATION, d_in=A5_D_IN, output_size=60, reference_family="depth",
        width_ladder=(16, 32, 64, 128), depth_max=6, hidden_size=16, learning_rate=0.01,
        n_epochs_cap=3000, patience=150, min_delta=1e-3, batch_size=64,
        moe_max_epochs=16000, moe_check_every=200, moe_patience=8, moe_min_delta=1e-3, moe_lr=0.01,
    ),
}


# ---------------------------------------------------------------------------
# Data builders — return (x_tr, y_tr, x_val, y_val, x_test, y_test) as float32/int64 numpy arrays,
# x always 2-D (N, d_in).
# ---------------------------------------------------------------------------


def _split_pool(n: int, seed: int, val_frac: float = VAL_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """Local shuffle-split of `range(n)` into (train_idx, val_idx) — a plain permutation, not `create_train_val_split`.

    `create_train_val_split` is built for `BaseModel.fit()`'s 3-way (train/val/test) partitioning;
    here the toys already supply their OWN held-out test set (an independent draw for the
    regression toys, `make_word_data`'s unseen-word half for A5), so only a 2-way split is needed
    — a permutation is the whole of what that requires.
    """
    idx = np.random.default_rng(seed).permutation(n)
    n_val = max(1, round(val_frac * n))
    return idx[n_val:], idx[:n_val]


def build_data(toy: ToyKey, seed: int) -> dict[str, np.ndarray]:
    """Builds (x_tr, y_tr, x_val, y_val, x_test, y_test) for `toy` at `seed` (see module docstring's split section)."""
    if toy is ToyKey.WIDTH:
        x_pool, y_pool, _region = nwn.make_hetero(1500, seed)
        x_te, y_te, _region_te = nwn.make_hetero(500, seed + 500)
        tr_idx, val_idx = _split_pool(len(x_pool), seed)
        return {
            "x_tr": x_pool[tr_idx].reshape(-1, 1), "y_tr": y_pool[tr_idx],
            "x_val": x_pool[val_idx].reshape(-1, 1), "y_val": y_pool[val_idx],
            "x_test": x_te.reshape(-1, 1), "y_test": y_te,
        }
    if toy is ToyKey.HETERO3:
        x_pool, y_pool, _region = nwn.make_hetero3(2250, seed)
        x_te, y_te, _region_te = nwn.make_hetero3(750, seed + 500)
        tr_idx, val_idx = _split_pool(len(x_pool), seed)
        return {
            "x_tr": x_pool[tr_idx].reshape(-1, 1), "y_tr": y_pool[tr_idx],
            "x_val": x_pool[val_idx].reshape(-1, 1), "y_val": y_pool[val_idx],
            "x_test": x_te.reshape(-1, 1), "y_test": y_te,
        }
    # A5_DEPTH
    data = dct.make_word_data(dct.Group.A5, seq_len=A5_SEQ_LEN, seed=seed)
    tr_idx, val_idx = _split_pool(len(data["x_tr"]), seed)
    return {
        "x_tr": data["x_tr"][tr_idx], "y_tr": data["y_tr"][tr_idx],
        "x_val": data["x_tr"][val_idx], "y_val": data["y_tr"][val_idx],
        "x_test": data["x_val"], "y_test": data["y_val"],  # make_word_data's OWN held-out unseen-word half
    }


# ---------------------------------------------------------------------------
# FlexNN-family training — bypass .fit()'s Optuna/early-stopping wrapper, call _fit_single
# directly, replay the trajectory through ConvergenceTracker (flexnn_revalidation.py pattern).
# ---------------------------------------------------------------------------


def _replay(val_loss_history: list[float], patience: int, min_delta: float) -> cvg.ConvergenceResult:
    tracker = cvg.ConvergenceTracker(patience=patience, min_delta=min_delta)
    for epoch, val in enumerate(val_loss_history, start=1):
        tracker.update(epoch, val)
    return tracker.result(final_epoch=len(val_loss_history))


def build_flex_hidden(cfg: ToyConfig, seed: int) -> FlexibleHiddenLayersNN:
    """The A5 toy's NATURAL dial net (depth); the OFF-dial contender on the two regression toys."""
    return FlexibleHiddenLayersNN(
        max_hidden_layers=cfg.depth_max, hidden_size=cfg.hidden_size, layer_selection_method=LayerSelectionMethod.SOFT_GATING,
        depth_regularization=DepthRegularization.NONE, n_predictor_layers=1, uncertainty_method=UncertaintyMethod.CONSTANT,
        task_type=cfg.task_type, output_size=cfg.output_size, learning_rate=cfg.learning_rate, n_epochs=cfg.n_epochs_cap,
        early_stopping_rounds=cfg.patience, batch_size=cfg.batch_size, random_seed=seed, calculate_feature_importance=False,
    )


def build_flex_width(cfg: ToyConfig, seed: int) -> FlexibleWidthNN:
    """The two regression toys' NATURAL dial net (width); the OFF-dial contender on the A5 toy."""
    return FlexibleWidthNN(
        widths=cfg.width_ladder, activation=ActivationFunction.RELU, width_selection_method=WidthSelectionMethod.NONE,
        task_type=cfg.task_type, output_size=cfg.output_size, learning_rate=cfg.learning_rate, n_epochs=cfg.n_epochs_cap,
        early_stopping_rounds=cfg.patience, batch_size=cfg.batch_size, random_seed=seed, calculate_feature_importance=False,
    )


def train_flex(model: FlexibleHiddenLayersNN | FlexibleWidthNN, data: dict[str, np.ndarray], cfg: ToyConfig) -> dict:
    """Direct `_fit_single` call + trajectory replay; returns the convergence verdict dict."""
    _best_epoch, val_loss_history = model._fit_single(data["x_tr"], data["y_tr"], x_val=data["x_val"], y_val=data["y_val"])
    hit_cap = len(val_loss_history) >= cfg.n_epochs_cap
    replay = _replay(val_loss_history, cfg.patience, cfg.min_delta)
    return {"epochs_trained": len(val_loss_history), "hit_cap": hit_cap, "trustworthy": bool(replay.trustworthy and not hit_cap), "convergence": replay.summary()}


# ---------------------------------------------------------------------------
# Raw-logit evaluation off the trained module directly (bypasses predict()'s decode, so both
# regression MSE and classification accuracy/CE come from the SAME code path across every
# contender and every fixed capacity in a family's ladder).
# ---------------------------------------------------------------------------


def _score(raw: torch.Tensor, y_true: np.ndarray, task_type: TaskType) -> dict[str, float]:
    if task_type is TaskType.REGRESSION:
        pred = raw[:, 0].detach().cpu().numpy()
        return {"mse": float(np.mean((pred - y_true) ** 2))}
    y_t = torch.as_tensor(y_true, dtype=torch.long)
    pred = raw.argmax(dim=1)
    acc = float((pred == y_t).float().mean().item())
    ce = float(nnf.cross_entropy(raw, y_t).item())
    return {"accuracy": acc, "ce": ce}


def _primary_metric(scored: dict[str, float], task_type: TaskType) -> float:
    """One scalar per contender for cross-contender comparison; HIGHER is better for both metrics used here (accuracy) after sign-flipping MSE."""
    return -scored["mse"] if task_type is TaskType.REGRESSION else scored["accuracy"]


def _flex_hidden_logits_at_depth(model: FlexibleHiddenLayersNN, x: np.ndarray, depth: int) -> torch.Tensor:
    x_t = torch.tensor(np.asarray(x, dtype=np.float32), device=model.device)
    model.model.eval()
    with torch.no_grad():
        return model.model.forward_at_depth(x_t, depth)


def _flex_width_logits_at_width(model: FlexibleWidthNN, x: np.ndarray, width: int) -> torch.Tensor:
    x_t = torch.tensor(np.asarray(x, dtype=np.float32), device=model.device)
    model.model.eval()
    with torch.no_grad():
        return model.model.forward_width(x_t, width)


def best_fixed_val_selected(
    model: FlexibleHiddenLayersNN | FlexibleWidthNN, family: str, ladder: tuple[int, ...], data: dict[str, np.ndarray], task_type: TaskType
) -> dict:
    """Val-selected best FIXED capacity, read off `model`'s already-trained joint weights (no retraining).

    Mirrors `fit_router`'s own `eval_fn` scoring convention (squared error / 0-1 error), applied
    globally (one capacity for the whole set) rather than per-sample. Reports the chosen
    capacity's TEST score plus its S2 accounting (`executed_flops` at that capacity; `params` is
    the SAME total footprint the MoE params-matched regime targets, since this reads off the
    SAME shared-weights net as the "routed" contender for this family, not a standalone net —
    stated explicitly rather than hand-waved).
    """
    logits_fn = _flex_hidden_logits_at_depth if family == "depth" else _flex_width_logits_at_width
    val_scores = {}
    for cap in ladder:
        raw = logits_fn(model, data["x_val"], cap)
        scored = _score(raw, data["y_val"], task_type)
        val_scores[cap] = _primary_metric(scored, task_type)
    best_cap = max(val_scores, key=val_scores.get)
    raw_test = logits_fn(model, data["x_test"], best_cap)
    test_scored = _score(raw_test, data["y_test"], task_type)
    params_total = ca.param_count(model.model, path_filter=ca.LOGVAR_HEAD_PATH_SUBSTRING)
    flops_at_cap = ca.executed_flops(model.model, best_cap)
    return {
        "chosen_capacity": best_cap, "val_scores_by_capacity": {str(k): v for k, v in val_scores.items()},
        "test_score": test_scored, "params_total_shared_net": params_total, "executed_flops_at_capacity": flops_at_cap,
        "params_note": "shares trunk/blocks+all heads with this family's routed contender; a standalone single-capacity deployment would need only its own head/blocks",
    }


# ---------------------------------------------------------------------------
# MoE training — full-batch Adam via convergence.fit_to_convergence (same gate depth_composition_
# toy.py's train_clf uses), closing over moe_regression.training_loss.
# ---------------------------------------------------------------------------


def train_moe(net: moe.MoERegressionNet, data: dict[str, np.ndarray], task: moe.Task, cfg: ToyConfig, alpha: float, seed: int) -> cvg.ConvergenceResult:
    """Full-batch Adam via `fit_to_convergence`, closing over `moe_regression.training_loss`."""
    torch.manual_seed(seed)
    device = "cpu"
    net.to(device)
    y_dtype = torch.float32 if task is moe.Task.MSE else torch.long
    x_tr = torch.as_tensor(data["x_tr"], dtype=torch.float32, device=device)
    y_tr = torch.as_tensor(data["y_tr"], dtype=y_dtype, device=device)
    x_val = torch.as_tensor(data["x_val"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(data["y_val"], dtype=y_dtype, device=device)
    if task is moe.Task.MSE:
        y_tr, y_val = y_tr.unsqueeze(1), y_val.unsqueeze(1)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.moe_lr)

    def step_fn() -> None:
        opt.zero_grad()
        route = net.route(x_tr)
        pred = net(x_tr, mode=moe.RoutingMode.TOP_K)
        loss, _ = moe.training_loss(pred, y_tr, route, net.n_experts, alpha=alpha, task=task)
        loss.backward()
        opt.step()

    def val_fn() -> float:
        net.eval()
        with torch.no_grad():
            route = net.route(x_val)
            pred = net(x_val, mode=moe.RoutingMode.TOP_K)
            loss, _ = moe.training_loss(pred, y_val, route, net.n_experts, alpha=alpha, task=task)
        net.train()
        return loss.item()

    result = cvg.fit_to_convergence(net, step_fn, val_fn, max_epochs=cfg.moe_max_epochs, check_every=cfg.moe_check_every, patience=cfg.moe_patience, min_delta=cfg.moe_min_delta)
    net.eval()
    return result


def eval_moe(net: moe.MoERegressionNet, data: dict[str, np.ndarray], task_type: TaskType) -> tuple[dict, dict]:
    """Test-set (score, routing_diagnostics) for a trained MoE net (TOP_K mode, deployment-accounting mode)."""
    x_test = torch.as_tensor(data["x_test"], dtype=torch.float32)
    net.eval()
    with torch.no_grad():
        route = net.route(x_test)
        raw = net(x_test, mode=moe.RoutingMode.TOP_K)
    scored = _score(raw, data["y_test"], task_type)
    diag = moe.routing_diagnostics(route, net.n_experts)
    return scored, diag


# ---------------------------------------------------------------------------
# One (toy, seed) cell — trains every contender, matches MoE in both regimes, applies the
# tuned-alpha rerun clause, lands the JSON.
# ---------------------------------------------------------------------------


def run_toy_seed(toy: ToyKey, seed: int) -> dict:
    """Trains every contender for one (toy, seed) cell, matches MoE in both regimes, lands the JSON."""
    cfg = TOY_CONFIGS[toy]
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = build_data(toy, seed)
    moe_task = moe.Task.MSE if cfg.task_type is TaskType.REGRESSION else moe.Task.CE

    flex_hidden = build_flex_hidden(cfg, seed)
    flex_hidden_conv = train_flex(flex_hidden, data, cfg)
    flex_width = build_flex_width(cfg, seed)
    flex_width_conv = train_flex(flex_width, data, cfg)

    flex_hidden.fit_router(data["x_val"], data["y_val"])
    flex_width.fit_router(data["x_val"], data["y_val"])
    pred_hidden_routed = flex_hidden.predict(data["x_test"], inference_mode="routed", filter_data=False)
    pred_width_routed = flex_width.predict(data["x_test"], inference_mode="routed", filter_data=False)

    def _decoded_score(pred: np.ndarray) -> dict[str, float]:
        if cfg.task_type is TaskType.REGRESSION:
            return {"mse": float(np.mean((pred - data["y_test"]) ** 2))}
        acc = float(np.mean(pred == data["y_test"]))
        return {"accuracy": acc}

    contenders: dict[str, dict] = {
        ContenderKind.FLEX_HIDDEN_ROUTED: {
            "test_score": _decoded_score(pred_hidden_routed),
            "mean_deployed_flops": flex_hidden.capacity_router_.mean_deployed_cost(data["x_test"]),
            "params_total": ca.param_count(flex_hidden.model),
            "convergence": flex_hidden_conv,
        },
        ContenderKind.FLEX_WIDTH_ROUTED: {
            "test_score": _decoded_score(pred_width_routed),
            "mean_deployed_flops": flex_width.capacity_router_.mean_deployed_cost(data["x_test"]),
            "params_total": ca.param_count(flex_width.model, path_filter=ca.LOGVAR_HEAD_PATH_SUBSTRING),
            "convergence": flex_width_conv,
        },
    }

    if cfg.reference_family == "width":
        reference_module, reference_config = flex_width.model, max(cfg.width_ladder)
        best_fixed = best_fixed_val_selected(flex_width, "width", cfg.width_ladder, data, cfg.task_type)
    else:
        reference_module, reference_config = flex_hidden.model, cfg.depth_max
        best_fixed = best_fixed_val_selected(flex_hidden, "depth", tuple(range(1, cfg.depth_max + 1)), data, cfg.task_type)
    contenders[ContenderKind.BEST_FIXED] = best_fixed

    match_info = moe.match_to_reference(reference_module, reference_config, n_experts=N_EXPERTS, top_k=TOP_K_PRIMARY, d_in=cfg.d_in, output_size=cfg.output_size)
    h_params = max(1, round(match_info["h_from_params_only"])) if match_info["h_from_params_only"] is not None else match_info["expert_hidden"]
    h_flops = max(1, round(match_info["h_from_flops_only"])) if match_info["h_from_flops_only"] is not None else match_info["expert_hidden"]
    regime_h = {MatchRegime.PARAMS: h_params, MatchRegime.FLOPS: h_flops}

    moe_results: dict[str, dict] = {}
    moe_nets: dict[str, moe.MoERegressionNet] = {}
    for regime, h in regime_h.items():
        for top_k, kind in ((TOP_K_PRIMARY, ContenderKind.MOE_TOP2), (TOP_K_ABLATION, ContenderKind.MOE_TOP1_ABLATION)):
            net = moe.MoERegressionNet(n_experts=N_EXPERTS, top_k=top_k, expert_hidden=h, d_in=cfg.d_in, output_size=cfg.output_size)
            result = train_moe(net, data, moe_task, cfg, ALPHA_FROZEN, seed)
            scored, diag = eval_moe(net, data, cfg.task_type)
            achieved_params, achieved_flops = moe.moe_flops_and_params(N_EXPERTS, h, cfg.d_in, cfg.output_size, top_k)
            key = f"{kind.value}__{regime.value}"
            moe_results[key] = {
                "regime": regime.value, "top_k": top_k, "expert_hidden": h, "alpha": ALPHA_FROZEN,
                "test_score": scored, "routing_diagnostics": diag, "achieved_params": achieved_params, "achieved_flops": achieved_flops,
                "trustworthy": bool(result.trustworthy), "convergence": result.summary(),
            }
            moe_nets[key] = net

    # Tuned-alpha rerun clause: does the frozen alpha=1e-2 run's best MoE variant lag the best
    # non-MoE contender by more than MOE_FAIL_MARGIN? If so, rerun THAT one config at two
    # bracketing alpha values and record whether tuning closes the gap (never a hand-wavy claim).
    non_moe_contenders = (contenders[ContenderKind.FLEX_HIDDEN_ROUTED], contenders[ContenderKind.FLEX_WIDTH_ROUTED], contenders[ContenderKind.BEST_FIXED])
    non_moe_best = max(_primary_metric(c["test_score"], cfg.task_type) for c in non_moe_contenders)
    moe_best_key = max(moe_results, key=lambda k: _primary_metric(moe_results[k]["test_score"], cfg.task_type))
    moe_best_metric = _primary_metric(moe_results[moe_best_key]["test_score"], cfg.task_type)
    relative_gap = (non_moe_best - moe_best_metric) / (abs(non_moe_best) + 1e-12)
    moe_underperforms = relative_gap > MOE_FAIL_MARGIN

    tuned_alpha_rerun: dict[str, Any] = {
        "triggered": moe_underperforms, "relative_gap_at_frozen_alpha": relative_gap,
        "moe_fail_margin": MOE_FAIL_MARGIN, "reran_config": moe_best_key, "results": {},
    }
    if moe_underperforms:
        best_cfg = moe_results[moe_best_key]
        for alpha in TUNED_ALPHA_CANDIDATES:
            net = moe.MoERegressionNet(
                n_experts=N_EXPERTS, top_k=best_cfg["top_k"], expert_hidden=best_cfg["expert_hidden"], d_in=cfg.d_in, output_size=cfg.output_size
            )
            result = train_moe(net, data, moe_task, cfg, alpha, seed)
            scored, _diag = eval_moe(net, data, cfg.task_type)
            gap_closed = _primary_metric(scored, cfg.task_type) >= non_moe_best
            tuned_alpha_rerun["results"][str(alpha)] = {"test_score": scored, "trustworthy": bool(result.trustworthy), "gap_closed": gap_closed}

    record = {
        "toy": toy.value, "seed": seed, "task_type": cfg.task_type.value, "d_in": cfg.d_in, "output_size": cfg.output_size,
        "n_train": int(data["x_tr"].shape[0]), "n_val": int(data["x_val"].shape[0]), "n_test": int(data["x_test"].shape[0]),
        "reference_family": cfg.reference_family, "reference_config": reference_config, "match_info": match_info,
        "match_regime_expert_hidden": {regime.value: h for regime, h in regime_h.items()},
        "contenders": contenders, "moe_variants": moe_results, "tuned_alpha_rerun": tuned_alpha_rerun,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"moe_comparison_{toy.value}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[moe_flexnn_comparison] wrote {out_path}")
    return record


# ---------------------------------------------------------------------------
# Selftest / smoke test -- tiny synthetic data, 1 seed, small epoch caps: proves the driver runs
# end-to-end and produces a well-formed JSON before committing to the full battery.
# ---------------------------------------------------------------------------


EXPECTED_MOE_VARIANT_COUNT = len(MatchRegime) * 2  # {top2, top1_ablation} x {params_matched, flops_matched}


def run_selftest() -> bool:
    """Fast smoke test: overrides every toy's config to a tiny/fast version, runs seed=0, checks the JSON."""
    tiny_overrides = {
        ToyKey.WIDTH: ToyConfig(
            key=ToyKey.WIDTH, task_type=TaskType.REGRESSION, d_in=1, output_size=1, reference_family="width",
            width_ladder=(2, 4), depth_max=2, hidden_size=4, learning_rate=0.02,
            n_epochs_cap=30, patience=5, min_delta=1e-4, batch_size=32,
            moe_max_epochs=200, moe_check_every=20, moe_patience=3, moe_min_delta=1e-4, moe_lr=0.02,
        ),
    }
    global TOY_CONFIGS  # noqa: PLW0603 -- selftest-local override, restored immediately after
    orig = TOY_CONFIGS
    TOY_CONFIGS = {**orig, **tiny_overrides}
    orig_make_hetero_n = None
    try:
        record = run_toy_seed(ToyKey.WIDTH, 0)
        ok_contenders = set(record["contenders"]) == {ContenderKind.FLEX_HIDDEN_ROUTED.value, ContenderKind.FLEX_WIDTH_ROUTED.value, ContenderKind.BEST_FIXED.value}
        ok_moe = len(record["moe_variants"]) == EXPECTED_MOE_VARIANT_COUNT
        ok_json = os.path.isfile(os.path.join(OUT_DIR, "moe_comparison_width_seed0.json"))
        ok = ok_contenders and ok_moe and ok_json
        print(
            f"[moe_flexnn_comparison selftest] contenders={sorted(record['contenders'])} "
            f"moe_variants={len(record['moe_variants'])} json_written={ok_json}  {'PASS' if ok else 'FAIL'}"
        )
        return ok
    finally:
        TOY_CONFIGS = orig
        del orig_make_hetero_n


def main() -> None:
    """Parses args and runs `--selftest` or the (toys x seeds) battery."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Fast smoke test: tiny config, 1 seed, confirms end-to-end JSON output.")
    parser.add_argument("--toys", type=str, default=",".join(k.value for k in ToyKey), help="Comma-separated toy keys to run.")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS_DEFAULT), help="Comma-separated seeds.")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    toys = [ToyKey(t) for t in args.toys.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    for toy in toys:
        for seed in seeds:
            run_toy_seed(toy, seed)


if __name__ == "__main__":
    main()
