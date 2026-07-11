"""H1 — ProbReg: two-phase schedule vs shipping joint gate vs fixed-k (per-input selector program, WS-C).

The decisive "does our model work" comparison, on the SAME shipping architecture
(`ProbabilisticRegressionNet`/`ProbabilisticRegressionModel`, driven directly from this
experiment script — no library edits, per L4/the H1 scope guard). Three arms, identical net
config (`input_size=1`, `SEPARATE_HEADS`, Gaussian-LTV, `REGRESSION_ONLY`, `k_max=6`), on toys
D/E + the C_broad control, 3 seeds:

  (a) SHIPPING JOINT — `NClassesSelectionMethod.SOFT_GATING` + `NClassesRegularization.ELBO`,
      fit the ordinary way (`ProbabilisticRegressionModel.fit`) — "the library's best combo,
      trained as today".
  (b) TWO-PHASE — phase 1 trains the classifier/regression heads/direct-regression head with a
      per-sample `k ~ Uniform{1..k_max}` masked-prefix schedule (mirrors
      `_capacity_ladder_nested.NestedKSurrogate.masked_prefix_nll`'s masking pattern and
      `ProbabilisticRegressionModel.predict_distribution`'s per-head-outputs read, applied to
      the REAL net), with the gate (`n_classes_predictor`) untouched ("quiescent"); phase 2
      freezes everything except `n_classes_predictor` and distills it, via ordinary soft-label
      cross-entropy, onto the S1-winning SOFT target (`capacity_ladder_k6._soft_targets`,
      per-tercile-EM-stacked responsibilities — S1/S2's certified recipe of record, RESULTS.md
      "## S1 —"/"## S2 —") built from the frozen phase-1 net's own held-out-within-train per-k
      scores. The gate's logit layout (k∈{2..k_max} then bypass LAST,
      `probabilistic_regression_net.py:66`) is respected via an explicit column permutation
      relative to the k-ascending `c_grid=[1..k_max]` convention `_soft_targets` uses.
  (c) FIXED-k SWEEP — the same net, `n_classes_selection_method=NONE`, fixed `n_classes=k` for
      k in 1..k_max, best k by held-out NLL.
  (d) OPTIONAL FINE-TUNE (measured, NO bar) — arm (b)'s net, unfrozen, 100 more epochs of
      ordinary joint training (the model's own `n_classes_strategy.forward` +
      `_calculate_custom_loss`, i.e. "as today" minus ELBO — see the PREREGISTRATION's
      implementation notes for why ELBO is left off here) at lr=1e-4. Prediction: a safe no-op
      (|ΔNLL| <= 0.01 nat). Explicitly droppable under budget pressure.

Primary metric (every arm): held-out NLL of the per-input BLENDED density —
`-mean_i logsumexp_k(log w_k(x_i) + log p_k(y_i | x_i))`, a two-level mixture (mixture over
k-modes, each mode itself a k-component `SEPARATE_HEADS` mixture) — computed by
`_per_k_log_density` (per-k component densities) + `_gate_probs_c_grid_ascending` (the gate's
own soft weights, reordered to match). G6 full-vector rule: every case also stores the FULL
selector marginal-probability vector (`marginal_p`, one entry per k-mode), never a collapsed
scalar (effective-k, argmax, ...).

Run `--selftest` before any real read (N=200, 50-epoch smoke case on toy D, all arms). Run
`--measure-one` to time ONE arm (b)/toy D/seed 0 unit at real settings and extrapolate the full
battery's wall-time (EXECUTION_PLAN.md §0b measure-one-unit-before-matrix convention, mirroring
T2's probe). Run with no flags for the full 3-toy x 3-seed matrix (ORCHESTRATOR-OWNED per the
plan's run-ownership rule — this script is authored to be launched by the orchestrator's
detached-run harness, not run interactively to completion by the authoring worker).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as nnf

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _toy_datasets as td  # noqa: E402
import _variational_em as vem  # noqa: E402 — reuse gaussian_log_density verbatim (matches _capacity_ladder_nested.py)
import capacity_ladder_k6 as ck6  # noqa: E402 — reuse _RouterMLP/_train_router/_soft_targets/_jsonable verbatim

from automl_package.enums import (  # noqa: E402
    NClassesRegularization,
    NClassesSelectionMethod,
    ProbabilisticRegressionOptimizationStrategy,
    ProbRegLossType,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel  # noqa: E402
from automl_package.utils.numerics import create_bins  # noqa: E402
from automl_package.utils.pytorch_utils import get_device  # noqa: E402

OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "H1")

SEEDS = (0, 1, 2)
TOYS = ("D", "E", "C_broad")
K_MAX = 6
C_GRID = list(range(1, K_MAX + 1))
SIG = 0.3
SEP_D = 4.0
SEP_MIN, SEP_MAX = 0.3, 4.0
N_TRAIN, N_TEST = 1500, 1500

# Shared net config (identical across arms a/b/c/d — only the training SCHEME differs).
NET_KW: dict = {
    "input_size": 1,
    "regression_strategy": RegressionStrategy.SEPARATE_HEADS,
    "uncertainty_method": UncertaintyMethod.PROBABILISTIC,
    "prob_reg_loss_type": ProbRegLossType.GAUSSIAN_LTV,
    "optimization_strategy": ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
    "constrain_middle_class": True,
    "use_monotonic_constraints": False,
    "use_anchored_heads": False,
    "calculate_feature_importance": False,
}
N_EPOCHS = 100  # arm (a)/(c) `.fit()` epoch budget; also phase-1's masked-prefix loop budget
LR = 1e-2
EARLY_STOPPING_ROUNDS = 15
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 64
FINETUNE_EPOCHS = 100  # arm (d)
FINETUNE_LR = 1e-4


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def _make_toy(toy: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generates one toy's (x, y), matching K4/S1's SIG/SEP_D/SEP_MIN/SEP_MAX conventions."""
    if toy == "D":
        return td.make_toy_d(n=n, sigma=SIG, separation=SEP_D, seed=seed)
    if toy == "E":
        return td.make_toy_e(n=n, sigma=SIG, sep_min=SEP_MIN, sep_max=SEP_MAX, seed=seed)
    if toy == "C_broad":
        return td.make_toy_c_broad(n=n, sigma=SIG, sep_min=SEP_MIN, sep_max=SEP_MAX, seed=seed)
    raise ValueError(f"unknown toy {toy!r}")


def _base_kwargs(n_epochs: int, seed: int) -> dict:
    """Shared `.fit()`-path kwargs for arms (a)/(c) — net config + the fitting-schedule constants."""
    return {
        **NET_KW,
        "n_epochs": n_epochs,
        "learning_rate": LR,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "validation_fraction": VALIDATION_FRACTION,
        "batch_size": BATCH_SIZE,
        "random_seed": seed,
    }


# ---------------------------------------------------------------------------
# The core reused primitive: per-k held-out log-density of the REAL net, c_grid-ascending.
# ---------------------------------------------------------------------------


def _per_k_log_density(net: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, k_max: int) -> torch.Tensor:
    """`(N, k_max)` held-out log-density table, columns ordered `c_grid=[1, 2, ..., k_max]`.

    Column 0 (k=1) is the direct/bypass single Gaussian (`net.direct_regression_head`, matching
    the whole capacity-ladder program's "k=1 rung = the bypass" convention, e.g.
    `_capacity_ladder_nested.py`'s docstring). Columns 1..k_max-1 (k=2..k_max) are the
    masked-prefix `SEPARATE_HEADS` mixture density over the first k classes — mirrors
    `_compute_predictions_for_k`'s masking (`probabilistic_regression_net.py:132`) and
    `ProbabilisticRegressionModel.predict_distribution`'s per-head-outputs read
    (`probabilistic_regression.py:604-618`), generalized across every prefix length k at once
    (same per-mode full-batch-then-select pattern `_weighted_average_logic` already uses,
    `base_selection_strategy.py:88-113` — no extra forward-pass cost vs the shipping model).

    Requires `net` to have been built with `n_classes_selection_method != NONE` (so
    `direct_regression_head`/`classifier_layers`/`regression_module` are sized to `k_max`); the
    gate (`n_classes_predictor`) itself is never touched here.
    """
    direct_out = net.direct_regression_head(x)  # (N, 2): mean, log_var
    col0 = vem.gaussian_log_density(y, direct_out[:, 0:1], direct_out[:, 1:2]).squeeze(1)

    classifier_raw_logits = net.classifier_layers(x)  # (N, k_max)
    cols = [col0]
    for k_val in range(2, k_max + 1):
        masked = torch.full_like(classifier_raw_logits, float("-inf"))
        masked[:, :k_val] = classifier_raw_logits[:, :k_val]
        probabilities = torch.softmax(masked, dim=1)
        _, per_head_outputs = net.regression_module(probabilities, return_head_outputs=True)  # (N, k_max, 2)
        log_w = nnf.log_softmax(masked, dim=1)  # (N, k_max), -inf outside the prefix
        log_phi = vem.gaussian_log_density(y, per_head_outputs[:, :, 0], per_head_outputs[:, :, 1])
        cols.append(torch.logsumexp(log_w + log_phi, dim=1))
    return torch.stack(cols, dim=1)


def _gate_probs_c_grid_ascending(net: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The gate's own softmax, reordered to match `_per_k_log_density`'s column order.

    Native layout is `[k=2..k_max, bypass]`; this returns `c_grid=[1(bypass), 2, ..., k_max]`.
    """
    probs = nnf.softmax(net.n_classes_predictor(x), dim=1)
    return torch.cat([probs[:, -1:], probs[:, :-1]], dim=1)


def _blended_nll(net: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, k_max: int) -> tuple[float, np.ndarray]:
    """Held-out NLL of the per-input blended density, plus the FULL gate selector vector.

    The marginal-probability vector honors the G6 full-vector rule: never a collapsed scalar
    summary.
    """
    net.eval()
    with torch.no_grad():
        log_dens = _per_k_log_density(net, x, y, k_max)
        probs = _gate_probs_c_grid_ascending(net, x)
        log_blend = torch.logsumexp(torch.log(probs.clamp_min(1e-300)) + log_dens, dim=1)
        nll = float(-log_blend.mean().item())
        marginal_p = probs.mean(dim=0).cpu().numpy().astype(np.float64)
    return nll, marginal_p


# ---------------------------------------------------------------------------
# Net-init bookkeeping arm (b) must replicate by hand (bypassing `.fit()`/`_fit_single`).
# ---------------------------------------------------------------------------


def _seed_head_centroids(model: ProbabilisticRegressionModel, y_train: np.ndarray, k_max: int) -> None:
    """Replicates the SEPARATE_HEADS centroid/middle-class-init bookkeeping arm (b) must redo by hand.

    `ProbabilisticRegressionModel._fit_single` normally does this before `build_model()`
    (`probabilistic_regression.py:519-530`) — so arm (b)'s net starts from the SAME
    initialization scheme as arms (a)/(c) even though its custom two-phase loop bypasses
    `.fit()` entirely. `precomputed_class_boundaries`/CE bookkeeping is intentionally NOT
    replicated: `optimization_strategy=REGRESSION_ONLY` + no boundary regularization + no
    middle-class penalty (our shared `NET_KW`) never reads it (see `_calculate_custom_loss`'s
    guards).
    """
    y_flat = y_train.ravel()
    _, y_binned = create_bins(data=y_flat, n_bins=k_max, min_value=-np.inf, max_value=np.inf)
    counts = np.bincount(y_binned, minlength=k_max)
    sums = np.bincount(y_binned, weights=y_flat, minlength=k_max)
    model._per_class_centroids = list(np.where(counts > 0, sums / counts, 0.0))
    if k_max % 2 == 1:
        mid_mask = y_binned == k_max // 2
        model._constant_head_init_value = float(y_flat[mid_mask].mean()) if mid_mask.any() else 0.0
    else:
        model._constant_head_init_value = None


# ---------------------------------------------------------------------------
# Arm (a): SHIPPING JOINT.
# ---------------------------------------------------------------------------


def _fit_shipping_joint(x_tr: np.ndarray, y_tr: np.ndarray, k_max: int, seed: int, n_epochs: int) -> ProbabilisticRegressionModel:
    model = ProbabilisticRegressionModel(
        **_base_kwargs(n_epochs, seed),
        n_classes=k_max,
        max_n_classes_for_probabilistic_path=k_max,
        n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING,
        n_classes_regularization=NClassesRegularization.ELBO,
    )
    model.fit(x_tr, y_tr)
    return model


# ---------------------------------------------------------------------------
# Arm (c): FIXED-k SWEEP.
# ---------------------------------------------------------------------------


def _fit_fixed_k(x_tr: np.ndarray, y_tr: np.ndarray, k: int, seed: int, n_epochs: int) -> ProbabilisticRegressionModel:
    model = ProbabilisticRegressionModel(
        **_base_kwargs(n_epochs, seed),
        n_classes=k,
        n_classes_selection_method=NClassesSelectionMethod.NONE,
        n_classes_regularization=NClassesRegularization.NONE,
    )
    model.fit(x_tr, y_tr)
    return model


def _eval_fixed_k_nll(model: ProbabilisticRegressionModel, x_te: np.ndarray, y_te: np.ndarray) -> float:
    """True mixture NLL via `predict_distribution` (not the LTV-collapsed Gaussian).

    Only valid for fixed `n_classes_selection_method=NONE`.
    """
    dist = model.predict_distribution(x_te)
    return float(-dist.log_prob(np.asarray(y_te).ravel()).mean())


# ---------------------------------------------------------------------------
# Arm (b), phase 1: masked-prefix schedule on the SAME net, gate quiescent.
# ---------------------------------------------------------------------------


def _build_phase1_host(y_tr: np.ndarray, k_max: int, seed: int, n_epochs: int) -> ProbabilisticRegressionModel:
    """Builds arm (b)'s net with the same config/init as arm (a), but does NOT fit it.

    Phase 1 drives the net directly with a custom loop below.
    """
    model = ProbabilisticRegressionModel(
        **_base_kwargs(n_epochs, seed),
        n_classes=k_max,
        max_n_classes_for_probabilistic_path=k_max,
        n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING,
        n_classes_regularization=NClassesRegularization.NONE,  # gate is quiescent in phase 1 — nothing to regularize
    )
    _seed_head_centroids(model, y_tr, k_max)
    # Replicates the seeding `PyTorchModelBase._fit_single` does right before `build_model()`
    # (base_pytorch.py:129-132), which arm (b) skips by not calling `.fit()`.
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.build_model()
    return model


def _phase1_masked_prefix_nll(net: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, k_draw: torch.Tensor, k_max: int) -> torch.Tensor:
    """Mean NLL of the per-sample drawn prefix `k_draw[i] in {1..k_max}`.

    The gate (`n_classes_predictor`) is never called, so it stays quiescent by construction.
    """
    log_dens = _per_k_log_density(net, x, y, k_max)  # (N, k_max), c_grid ascending, grad-enabled
    col = (k_draw - 1).unsqueeze(1)  # k=1 -> col 0, ..., k=k_max -> col k_max-1
    picked = log_dens.gather(1, col).squeeze(1)
    return -picked.mean()


def _train_phase1(net: torch.nn.Module, x_t: torch.Tensor, y_t: torch.Tensor, k_max: int, n_epochs: int, lr: float, seed: int) -> None:
    """Full-batch Adam over classifier_layers + regression_module + direct_regression_head only.

    A fresh per-epoch `k ~ Uniform{1..k_max}` draw — mirrors
    `_capacity_ladder_nested.train_nested_k_surrogate`'s training-loop shape.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    params = list(net.classifier_layers.parameters()) + list(net.regression_module.parameters()) + list(net.direct_regression_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    n = x_t.shape[0]
    net.train()
    for _ in range(int(n_epochs)):
        opt.zero_grad()
        k_draw = (torch.randint(1, k_max + 1, (n,), generator=gen)).to(x_t.device)
        loss = _phase1_masked_prefix_nll(net, x_t, y_t, k_draw, k_max)
        loss.backward()
        opt.step()
    net.eval()


# ---------------------------------------------------------------------------
# Arm (b), phase 2: freeze everything except n_classes_predictor, distill the S1-SOFT target.
# ---------------------------------------------------------------------------


def _phase2_build_targets(net: torch.nn.Module, x_t: torch.Tensor, y_t: torch.Tensor, k_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Held-out-within-train score table from the FROZEN phase-1 net, and the S1-SOFT target.

    Built via `capacity_ladder_k6._soft_targets` (per-tercile EM-stacked responsibilities —
    the certified recipe of record, RESULTS.md "## S1 —"). Returns
    `(q_gate, q_ascending, x_np)`: `q_gate` is reordered to the real gate's `[k=2..k_max,
    bypass]` layout (`probabilistic_regression_net.py:66`); `q_ascending` is the untouched
    `c_grid=[1..k_max]`-ordered target, reused verbatim for the standalone-router parity arm
    (bar iii) so both hosts train on the IDENTICAL target.
    """
    net.eval()
    with torch.no_grad():
        score = _per_k_log_density(net, x_t, y_t, k_max).cpu().numpy().astype(np.float64)  # c_grid ascending
    x_np = x_t.detach().cpu().numpy().ravel().astype(np.float64)
    q_ascending = ck6._soft_targets(score, x_np)
    q_gate = np.concatenate([q_ascending[:, 1:], q_ascending[:, 0:1]], axis=1)  # k=2..k_max, bypass LAST
    return q_gate, q_ascending, x_np


def _train_phase2_gate(net: torch.nn.Module, x_t: torch.Tensor, q_gate: np.ndarray, n_epochs: int, lr: float, seed: int, device: str) -> None:
    """Soft-label cross-entropy to `q_gate`, over `net.n_classes_predictor.parameters()` ONLY.

    The optimizer's own parameter set is the freeze mechanism (the frozen submodules are never
    forwarded through here at all, so no gradient could reach them regardless).
    """
    torch.manual_seed(seed)
    q_t = torch.as_tensor(q_gate, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(net.n_classes_predictor.parameters(), lr=lr)
    net.n_classes_predictor.train()
    for _ in range(int(n_epochs)):
        opt.zero_grad()
        logits = net.n_classes_predictor(x_t)
        loss = -(q_t * nnf.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss.backward()
        opt.step()
    net.n_classes_predictor.eval()


# ---------------------------------------------------------------------------
# Bar (iii): standalone S1-SOFT router (`capacity_ladder_k6._RouterMLP`) — same target, same
# frozen density source, different host.
# ---------------------------------------------------------------------------


def _standalone_router_nll(
    net: torch.nn.Module, x_p2_np: np.ndarray, q_ascending: np.ndarray, x_te_t: torch.Tensor, y_te_t: torch.Tensor, k_max: int, device: str, seed: int,
) -> float:
    """Blend NLL of a standalone `ck6._RouterMLP` trained on `q_ascending`, evaluated on `x_te_t`."""
    router = ck6._train_router(x_p2_np, n_cols=k_max, device=device, soft_targets=q_ascending, seed=seed, n_epochs=ck6.N_EPOCHS, lr=ck6.LR)
    with torch.no_grad():
        score_te = _per_k_log_density(net, x_te_t, y_te_t, k_max)  # frozen phase-1 net, c_grid ascending
        router_logits = router(x_te_t)
        log_w = nnf.log_softmax(router_logits, dim=1)
        nll = float(-torch.logsumexp(log_w + score_te, dim=1).mean().item())
    del router
    return nll


# ---------------------------------------------------------------------------
# Arm (d): OPTIONAL FINE-TUNE (measured, NO bar) — arm (b) + joint training, unfrozen.
# ---------------------------------------------------------------------------


def _finetune_joint(model: ProbabilisticRegressionModel, net: torch.nn.Module, x_tr: np.ndarray, y_tr: np.ndarray, n_epochs: int, lr: float, seed: int, device: str) -> None:
    """Resumes arm (b)'s distilled net with ordinary joint training, unfrozen.

    Uses the model's own native forward (`n_classes_strategy.forward`, i.e. `SoftGatingStrategy`)
    + native loss (`model._calculate_custom_loss`, which includes the library's auto-gated
    ordering constraint for this SEPARATE_HEADS/Gaussian-LTV/REGRESSION_ONLY combo — "as today").

    Judgment call: `model.n_classes_regularization` is left at whatever the caller set
    (this script sets it to NONE before calling) rather than switching ELBO back on — arm (d)
    tests whether unfreezing everything under the plain NLL is a safe no-op relative to arm
    (b), not whether re-introducing the ELBO prior perturbs the already-distilled gate. See the
    PREREGISTRATION's implementation notes.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    x_t = torch.as_tensor(x_tr, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_tr, dtype=torch.float32, device=device).reshape(-1, 1)
    net.train()
    for _ in range(int(n_epochs)):
        opt.zero_grad()
        outputs = net(x_t)
        loss = model._calculate_custom_loss(outputs, y_t)
        loss.backward()
        opt.step()
    net.eval()


# ---------------------------------------------------------------------------
# One case (one toy/seed): all four arms + the standalone-router parity check.
# ---------------------------------------------------------------------------


def run_case(toy: str, seed: int, k_max: int, device: str, *, n_train: int, n_test: int, n_epochs: int, finetune_epochs: int) -> dict:
    """Runs all four arms + the standalone-router parity check on one (toy, seed) case."""
    x_tr, y_tr = _make_toy(toy, n_train, seed)
    x_te, y_te = _make_toy(toy, n_test, seed + 500)
    x_te_t = torch.as_tensor(x_te, dtype=torch.float32, device=device)
    y_te_t = torch.as_tensor(y_te, dtype=torch.float32, device=device)

    result: dict = {"toy": toy, "seed": seed, "n_train": n_train, "n_test": n_test}

    # ---- Arm (a): SHIPPING JOINT ----
    model_a = _fit_shipping_joint(x_tr, y_tr, k_max, seed, n_epochs)
    nll_a, marg_a = _blended_nll(model_a.model, x_te_t, y_te_t, k_max)
    result["a_shipping_joint"] = {"nll_blend": nll_a, "marginal_p": marg_a.tolist()}
    del model_a
    _release_memory()

    # ---- Arm (b): TWO-PHASE ----
    model_b = _build_phase1_host(y_tr, k_max, seed, n_epochs)
    net_b = model_b.model

    # phase 1 fits on one half of x_train; phase 2's targets are built on the OTHER half (the
    # "held-out-within-train split" the plan requires) — deterministic index-parity split, no
    # unseeded RNG, matching S1/K6's convention.
    p1_idx = np.arange(0, len(x_tr), 2)
    p2_idx = np.arange(1, len(x_tr), 2)
    x_p1 = torch.as_tensor(x_tr[p1_idx], dtype=torch.float32, device=device)
    y_p1 = torch.as_tensor(y_tr[p1_idx], dtype=torch.float32, device=device)
    x_p2 = torch.as_tensor(x_tr[p2_idx], dtype=torch.float32, device=device)
    y_p2 = torch.as_tensor(y_tr[p2_idx], dtype=torch.float32, device=device)

    _train_phase1(net_b, x_p1, y_p1, k_max, n_epochs, LR, seed)
    q_gate, q_ascending, x_p2_np = _phase2_build_targets(net_b, x_p2, y_p2, k_max)
    _train_phase2_gate(net_b, x_p2, q_gate, ck6.N_EPOCHS, ck6.LR, seed, device)

    nll_b, marg_b = _blended_nll(net_b, x_te_t, y_te_t, k_max)
    result["b_two_phase"] = {"nll_blend": nll_b, "marginal_p": marg_b.tolist()}

    # ---- bar (iii): standalone S1-SOFT router parity (same target, same frozen density source) ----
    nll_b_router = _standalone_router_nll(net_b, x_p2_np, q_ascending, x_te_t, y_te_t, k_max, device, seed)
    result["b_standalone_router_nll_blend"] = nll_b_router

    # ---- Arm (d): OPTIONAL FINE-TUNE (measured, no bar) — reuses arm (b)'s net, then discards it ----
    model_b.n_classes_regularization = NClassesRegularization.NONE
    _finetune_joint(model_b, net_b, x_tr, y_tr, finetune_epochs, FINETUNE_LR, seed, device)
    nll_d, marg_d = _blended_nll(net_b, x_te_t, y_te_t, k_max)
    result["d_finetune"] = {"nll_blend": nll_d, "marginal_p": marg_d.tolist(), "delta_vs_b": nll_d - nll_b}
    del model_b, net_b
    _release_memory()

    # ---- Arm (c): FIXED-k SWEEP ----
    per_k_nll: dict[int, float] = {}
    for k in range(1, k_max + 1):
        model_c = _fit_fixed_k(x_tr, y_tr, k, seed, n_epochs)
        per_k_nll[k] = _eval_fixed_k_nll(model_c, x_te, y_te)
        del model_c
        _release_memory()
    best_k = min(per_k_nll, key=per_k_nll.get)
    result["c_fixed_k"] = {"per_k_nll": per_k_nll, "best_k": best_k, "best_nll": per_k_nll[best_k]}

    return result


# ---------------------------------------------------------------------------
# Pre-registered bars (i)-(iv).
# ---------------------------------------------------------------------------


def _bar_i(cases: list[dict]) -> dict:
    """(i) (b) >= (a) on toy D held-out NLL, 3/3 seeds — i.e. `nll_b <= nll_a` (b's NLL not worse)."""
    d_cases = [c for c in cases if c["toy"] == "D"]
    n_pass = sum(1 for c in d_cases if c["b_two_phase"]["nll_blend"] <= c["a_shipping_joint"]["nll_blend"])
    return {"n_pass": n_pass, "n_cases": len(d_cases), "pass": bool(d_cases) and n_pass == len(d_cases)}


def _bar_ii(cases: list[dict]) -> dict:
    """(ii) (b) within 0.02 nat of best (c) on C_broad.

    Read as a two-sided |diff|, matching bar (iii)'s own "|ΔNLL| <= 0.01" formula in the same
    plan sentence structure.
    """
    broad_cases = [c for c in cases if c["toy"] == "C_broad"]
    diffs = [abs(c["b_two_phase"]["nll_blend"] - c["c_fixed_k"]["best_nll"]) for c in broad_cases]
    return {"diffs": diffs, "n_cases": len(broad_cases), "pass": bool(broad_cases) and all(d <= 0.02 for d in diffs)}


def _bar_iii(cases: list[dict]) -> dict:
    """(iii) parity: (b)'s phase-2 gate ~= the standalone S1-SOFT router, |ΔNLL| <= 0.01, all cases."""
    diffs = [abs(c["b_two_phase"]["nll_blend"] - c["b_standalone_router_nll_blend"]) for c in cases]
    n_pass = sum(1 for d in diffs if d <= 0.01)
    return {"diffs": diffs, "n_cases": len(cases), "n_pass": n_pass, "pass": bool(cases) and n_pass == len(cases)}


def _bar_iv_report(cases: list[dict]) -> list[dict]:
    """(iv) toy E: report only, NO bar (T3 owns the moving-mode question)."""
    return [c for c in cases if c["toy"] == "E"]


# ---------------------------------------------------------------------------
# Selftest — tiny toy-D subsample, all arms end-to-end, in-memory (real toy generator, not synthetic).
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Runs a tiny toy-D case (N=200, 50 ep) through all arms and checks every output is finite."""
    device = str(get_device())
    print(f"H1 selftest (N_train=200, N_test=200, epochs=50, k_max={K_MAX}, device={device})")
    case = run_case("D", seed=0, k_max=K_MAX, device=device, n_train=200, n_test=200, n_epochs=50, finetune_epochs=10)

    checks: list[tuple[str, bool]] = []
    for key in ("a_shipping_joint", "b_two_phase", "d_finetune"):
        nll = case[key]["nll_blend"]
        checks.append((f"{key} nll_blend finite ({nll:.4f})", math.isfinite(nll)))
        marg = case[key]["marginal_p"]
        ok_vec = len(marg) == K_MAX and all(math.isfinite(v) for v in marg) and abs(sum(marg) - 1.0) < 1e-3
        checks.append((f"{key} marginal_p full vector len={len(marg)} sum={sum(marg):.4f}", ok_vec))

    checks.append((f"b standalone router nll_blend finite ({case['b_standalone_router_nll_blend']:.4f})", math.isfinite(case["b_standalone_router_nll_blend"])))

    per_k = case["c_fixed_k"]["per_k_nll"]
    ok_c = len(per_k) == K_MAX and all(math.isfinite(v) for v in per_k.values())
    checks.append((f"c fixed-k: all {K_MAX} NLLs finite, best_k={case['c_fixed_k']['best_k']}", ok_c))

    for msg, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    all_pass = all(ok for _, ok in checks)
    print("all checks passed" if all_pass else "FAILURES PRESENT")
    return all_pass


# ---------------------------------------------------------------------------
# Real read: the full toy x seed matrix, bars, summary json.
# ---------------------------------------------------------------------------


def main() -> None:
    """Reads/generates the full toy x seed matrix, runs the pre-registered bars, writes the summary json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Run the H1 tiny-subsample selftest and exit.")
    parser.add_argument(
        "--measure-one", action="store_true", help="Time ONE arm-b (b_two_phase, toy D, seed=0) unit at real settings, print the extrapolated battery wall-time, and exit."
    )
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    device = str(get_device())

    if args.measure_one:
        toy, seed = TOYS[0], 0
        print(f"[h1] measure-one: arm=b_two_phase seed={seed} toy={toy} device={device} N_TRAIN={N_TRAIN} N_TEST={N_TEST} epochs={N_EPOCHS} k_max={K_MAX}")
        t0 = time.time()

        x_tr, y_tr = _make_toy(toy, N_TRAIN, seed)
        x_te, y_te = _make_toy(toy, N_TEST, seed + 500)
        x_te_t = torch.as_tensor(x_te, dtype=torch.float32, device=device)
        y_te_t = torch.as_tensor(y_te, dtype=torch.float32, device=device)

        # Exact arm-(b) wiring `run_case` uses (`run_case`'s `# ---- Arm (b): TWO-PHASE ----`
        # block) — reuses the existing arm-b functions/constants verbatim, no training-logic
        # duplication.
        model_b = _build_phase1_host(y_tr, K_MAX, seed, N_EPOCHS)
        net_b = model_b.model

        p1_idx = np.arange(0, len(x_tr), 2)
        p2_idx = np.arange(1, len(x_tr), 2)
        x_p1 = torch.as_tensor(x_tr[p1_idx], dtype=torch.float32, device=device)
        y_p1 = torch.as_tensor(y_tr[p1_idx], dtype=torch.float32, device=device)
        x_p2 = torch.as_tensor(x_tr[p2_idx], dtype=torch.float32, device=device)
        y_p2 = torch.as_tensor(y_tr[p2_idx], dtype=torch.float32, device=device)

        _train_phase1(net_b, x_p1, y_p1, K_MAX, N_EPOCHS, LR, seed)
        q_gate, _q_ascending, _x_p2_np = _phase2_build_targets(net_b, x_p2, y_p2, K_MAX)
        _train_phase2_gate(net_b, x_p2, q_gate, ck6.N_EPOCHS, ck6.LR, seed, device)
        _blended_nll(net_b, x_te_t, y_te_t, K_MAX)  # matches run_case's arm-(b) unit exactly, incl. the eval pass

        wall = time.time() - t0
        n_arms = 4  # a_shipping_joint, b_two_phase, c_fixed_k (sweep counted as one arm), d_finetune
        n_units = n_arms * len(TOYS) * len(SEEDS)
        print(f"[h1] one unit wall-time: {wall:.1f}s ({wall / 60.0:.2f} min)")
        print(f"[h1] extrapolated {n_units}-unit battery (serialized): {wall * n_units:.0f}s ({wall * n_units / 3600.0:.2f} h)")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[h1] device={device}")

    cases: list[dict] = []
    for toy in TOYS:
        for seed in SEEDS:
            print(f"[h1] {toy} s{seed} ...", flush=True)
            case = run_case(toy, seed, K_MAX, device, n_train=N_TRAIN, n_test=N_TEST, n_epochs=N_EPOCHS, finetune_epochs=FINETUNE_EPOCHS)
            cases.append(case)
            print(
                f"[h1] {toy} s{seed}: a={case['a_shipping_joint']['nll_blend']:.4f} b={case['b_two_phase']['nll_blend']:.4f} "
                f"b_router={case['b_standalone_router_nll_blend']:.4f} d={case['d_finetune']['nll_blend']:.4f} "
                f"c_best(k={case['c_fixed_k']['best_k']})={case['c_fixed_k']['best_nll']:.4f}"
            )

    bars = {
        "i_b_ge_a_on_D": _bar_i(cases),
        "ii_b_within_0.02_of_best_c_on_Cbroad": _bar_ii(cases),
        "iii_parity_gate_vs_standalone_router": _bar_iii(cases),
        "iv_toy_E_report_only": _bar_iv_report(cases),
    }
    for name, res in bars.items():
        print(f"[h1] bar {name}: {res}")

    summary = {
        "config": {
            "k_max": K_MAX,
            "c_grid": C_GRID,
            "toys": list(TOYS),
            "seeds": list(SEEDS),
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "n_epochs": N_EPOCHS,
            "lr": LR,
            "phase2_epochs": ck6.N_EPOCHS,
            "phase2_lr": ck6.LR,
            "finetune_epochs": FINETUNE_EPOCHS,
            "finetune_lr": FINETUNE_LR,
        },
        "cases": cases,
        "bars": bars,
    }
    out_path = os.path.join(OUT_DIR, "h1_summary.json")
    with open(out_path, "w") as f:
        json.dump(ck6._jsonable(summary), f, indent=2)
    print(f"[h1] wrote {out_path}")


if __name__ == "__main__":
    main()
