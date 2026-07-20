"""M2 — realistic toy mixture-of-experts (MoE) baseline, per the frozen M1 config.

`docs/plans/capacity_programme/flexnn-moe.md` Task M2, built against the "Frozen MoE config"
block that task M1 verified against primary sources (2026-07-16):

  * **Experts & routing — primary: 8 experts, top-2** (Mixtral, Jiang et al. 2024,
    arXiv:2401.04088, §2.1 / Table 1). **Ablation: top-1** (Switch Transformer, Fedus, Zoph &
    Shazeer 2022, arXiv:2101.03961, §2).
  * **Gating — deterministic top-k softmax, NO noise**: `G(x) = Softmax(TopK(x*W_g))`,
    `y = sum_i G(x)_i * Expert_i(x)` (Mixtral §2.1). Noisy top-k gating (Shazeer et al. 2017,
    arXiv:1701.06538 §2.1) is that paper's own mechanism only — not used here.
  * **Load-balance auxiliary loss** (Switch §2.2 eq. 4): `L_aux = alpha * E * sum_e f_e * P_e`,
    `alpha = 1e-2`. This module's `load_balance_aux_loss` returns the FULLY SCALED term (`alpha`
    already applied, matching the paper's eq. 4 verbatim) — the total training objective is
    `MSE + L_aux`, with no second `alpha` multiply (see `training_loss`'s docstring).

Experts are small MLPs `Linear(d_in -> expert_hidden) -> tanh -> Linear(expert_hidden ->
output_size)`; the router is one `Linear(d_in -> n_experts)`. Params/FLOPs accounting is NEVER
hand-computed here for the M2-frozen `output_size=1` (MSE) shape —
`match_to_reference` calls `capacity_accounting.param_count`/`executed_flops` (S2,
`docs/plans/capacity_programme/shared/metrics-accounting.md`), the programme's only source of
those numbers, for both the reference net and (via `MoEShapeDescriptor`) this module's own
architecture.

**F6 `task={mse,ce}` flag** (`docs/plans/capacity_programme/flexnn-core.md` Task F6): `Task.CE`
generalizes each expert's output layer to `Linear(expert_hidden -> n_classes)` (`output_size`
constructor arg) and `training_loss` to `CrossEntropyLoss + L_aux` on the SAME router/aux-loss
machinery — needed because F6's A5 depth-composition toy is classification, not regression.
`capacity_accounting.MoEShapeDescriptor` (S2) hardcodes a scalar (`output_size=1`) per-expert
output layer — the M2-authoring-time shape, since S2 predates this flag — so it cannot account a
CE-task MoE's true expert output-layer width. Extending `MoEShapeDescriptor` itself is out of
this task's file list (`capacity_accounting.py` is owned by a concurrent capacity-programme task
this session). `moe_flops_and_params()` below is a same-shape generalization CONFINED to this
module: for `output_size=1` it is proven byte-identical to S2's own formula (selftest (e)), so
`match_to_reference` keeps calling S2 unchanged for `task=mse` (S2 remains the only source for
the shape it supports) and only falls back to the local generalized formula for `task=ce`.

Selftest (`--selftest`, mirrors `nested_width_net.py`'s selftest convention) proves four things:
  (a) shapes/finiteness of both routing modes' forward output.
  (b) top-k mask really zeroes non-selected experts' gradient paths: perturbing a non-selected
      expert's weights leaves the output UNCHANGED for every row that didn't select it (the
      `nested_width_net.py` selftest-(a) prefix-invariance pattern, applied to expert dispatch
      instead of width prefixes).
  (c) the load-balance aux loss decreases under gradient descent starting from a deliberately
      imbalanced router.
  (d) `match_to_reference` hits <=5% relative error on BOTH total params and executed FLOPs for
      two reference configs from two different families (FlexNN, a width net).

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/moe_regression.py --selftest
"""

from __future__ import annotations

import argparse
import enum
import os
import sys
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as nnf

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import capacity_accounting as ca  # noqa: E402
import nested_width_net as nwn  # noqa: E402 — selftest's width-net reference config only

from automl_package.enums import LayerSelectionMethod  # noqa: E402 — selftest's FlexNN reference config only
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN  # noqa: E402

# --- Frozen M1 config (module docstring) ------------------------------------------------------
N_EXPERTS_PRIMARY = 8
TOP_K_PRIMARY = 2  # Mixtral
TOP_K_ABLATION = 1  # Switch
ALPHA = 1e-2  # Switch eq. 4
D_IN_DEFAULT = 1  # toy scalar input, matching the width/FlexNN nets this MoE is compared against
EXPERT_HIDDEN_DEFAULT = 8

MATCH_TOLERANCE = 0.05  # M2 brief: match_to_reference must hit <=5% on params AND FLOPs
_MATCH_REFINE_WINDOW = 25  # integer search radius around each closed-form h solution

_COLLAPSE_LOAD_DIVISOR = 4  # M2 brief: an expert is "collapsed" if its load < 1/(4*n_experts)
_TOPK_ISOLATION_TOL = 1e-6


class RoutingMode(enum.Enum):
    """Which weight vector `MoERegressionNet.forward` uses (closed set, not a string literal)."""

    TOP_K = "top_k"  # deployment/accounting mode: executed = top_k experts (capacity_accounting.MoEShapeDescriptor)
    FULL_SOFT = "full_soft"  # diagnostic: every expert contributes, weighted by the full (unmasked) softmax


class Task(enum.StrEnum):
    """Which task `MoERegressionNet`/`training_loss` is wired for (closed set; F6 flag)."""

    MSE = "mse"  # M2's original frozen shape: output_size=1, MSELoss + L_aux
    CE = "ce"  # F6 addition: output_size=n_classes, CrossEntropyLoss + L_aux (A5 depth-composition toy)


class RouterOutputs(NamedTuple):
    """One `route()` call's full router state, reused by `forward`, the aux loss, and diagnostics."""

    router_logits: torch.Tensor  # (N, E) raw router logits, pre-softmax
    full_probs: torch.Tensor  # (N, E) softmax over ALL experts (no top-k masking) — Switch's p_i(x)
    gate_probs: torch.Tensor  # (N, E) softmax(TopK(logits)) == Mixtral's G(x); zero off the selected top-k
    dispatch_mask: torch.Tensor  # (N, E) 1.0 where expert e is in that row's top-k set, else 0.0
    topk_idx: torch.Tensor  # (N, top_k) int64 indices of the selected experts per row


class MoERegressionNet(nn.Module):
    """Toy top-k MoE regressor: `n_experts` small MLP experts + one linear router.

    Each expert is `Linear(d_in -> expert_hidden) -> tanh -> Linear(expert_hidden -> 1)`; the
    router is `Linear(d_in -> n_experts)` (no noise — deterministic top-k softmax gating, Mixtral
    §2.1, per the module docstring's frozen config).
    """

    def __init__(
        self,
        n_experts: int = N_EXPERTS_PRIMARY,
        top_k: int = TOP_K_PRIMARY,
        expert_hidden: int = EXPERT_HIDDEN_DEFAULT,
        d_in: int = D_IN_DEFAULT,
        output_size: int = 1,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        """Builds the router and `n_experts` expert MLPs.

        Args:
            n_experts: total expert count `E` (router output width; every expert's params exist
                regardless of `top_k`).
            top_k: experts dispatched to per row (2 = Mixtral primary, 1 = Switch ablation).
            expert_hidden: each expert's hidden width — the ONE knob `match_to_reference` sizes.
            d_in: input dimensionality (1 for this programme's scalar-`x` toys).
            output_size: each expert's output width — `1` for the M2-frozen MSE shape, `n_classes`
                for `Task.CE` (F6 flag; see module docstring).
            activation: expert hidden-layer nonlinearity class (instantiated with no args); tanh
                per the frozen spec, matching this programme's other toy nets.

        Raises:
            ValueError: `top_k` is not in `[1, n_experts]`.
        """
        super().__init__()
        if not (1 <= top_k <= n_experts):
            raise ValueError(f"top_k={top_k} out of range [1, {n_experts}]")
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.d_in = int(d_in)
        self.output_size = int(output_size)
        self.router = nn.Linear(self.d_in, self.n_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(self.d_in, expert_hidden), activation(), nn.Linear(expert_hidden, self.output_size)) for _ in range(self.n_experts)
        )

    def route(self, x: torch.Tensor) -> RouterOutputs:
        """Computes router logits, both probability views, and the dispatch mask for `x`.

        `full_probs` is the plain softmax over every expert (Switch's `p_i(x)`, eq. 4) — used for
        the load-balance aux loss's `P_e` term and as `FULL_SOFT` mode's forward weights.
        `gate_probs` is `G(x) = Softmax(TopK(logits))` (Mixtral §2.1) — the sparse, renormalized
        weight vector used for `TOP_K` mode's forward: `TopK` sets every non-selected logit to
        `-inf` before the softmax, so `gate_probs` is EXACTLY zero off the selected experts (not
        an approximation — `exp(-inf) == 0.0` in IEEE754), which is what makes selftest (b)'s
        gradient-isolation check exact rather than approximate.
        """
        logits = self.router(x)  # (N, n_experts)
        full_probs = torch.softmax(logits, dim=-1)
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(-1, topk_idx, topk_vals)
        gate_probs = torch.softmax(masked_logits, dim=-1)
        dispatch_mask = torch.zeros_like(logits)
        dispatch_mask.scatter_(-1, topk_idx, 1.0)
        return RouterOutputs(router_logits=logits, full_probs=full_probs, gate_probs=gate_probs, dispatch_mask=dispatch_mask, topk_idx=topk_idx)

    def forward(self, x: torch.Tensor, *, mode: RoutingMode = RoutingMode.TOP_K) -> torch.Tensor:
        """`(N, d_in) -> (N, output_size)` prediction under the given routing mode.

        `RoutingMode.TOP_K` is the deployment/accounting mode (`capacity_accounting.executed_flops`
        assumes exactly `top_k` experts run). `RoutingMode.FULL_SOFT` is a diagnostic dense upper
        bound that additionally weights every non-selected expert by its full-softmax probability
        — never used for the executed-FLOPs comparison, only to characterize what a dense soft-MoE
        would do. Both modes run every expert's forward pass densely (this module's non-goal is
        sparse kernels — dense masked compute is fine at toy scale, `executed_flops` accounts for
        the EFFICIENT top-k deployment analytically regardless of how this eval code computes it,
        mirroring how `nested_width_net.py`'s `all_widths_forward` also computes densely).

        Stacks (not concatenates) each expert's `(N, output_size)` output into `(N, n_experts,
        output_size)` and weights along the expert axis — the `output_size=1` (MSE) case reduces
        to M2's original `cat`+`sum(keepdim=True)` behavior exactly (same numbers, generalized
        shape), and `output_size=n_classes` (F6 `Task.CE`) gets the identical weighted-sum-of-
        experts semantics over class logits.
        """
        route = self.route(x)
        weights = route.gate_probs if mode is RoutingMode.TOP_K else route.full_probs
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=1)  # (N, n_experts, output_size)
        return (weights.unsqueeze(-1) * expert_out).sum(dim=1)  # (N, output_size)


def load_balance_aux_loss(route: RouterOutputs, n_experts: int, alpha: float = ALPHA) -> torch.Tensor:
    """Switch Transformer eq. 4 load-balance loss: `alpha * E * sum_e f_e * P_e` (alpha baked in).

    `f_e` = fraction of the batch DISPATCHED to expert e (`dispatch_mask` column mean) —
    generalizes Switch's top-1 `f_i` to top-k>1 by counting every dispatch pair, so `sum_e f_e ==
    top_k` rather than 1. This top-k generalization is the natural extension of eq. 4, not itself
    a cited claim: Mixtral's paper does not restate an aux loss (see the "Frozen MoE config" block
    in `docs/plans/capacity_programme/flexnn-moe.md`). `P_e` = mean of the FULL (unmasked) softmax
    probability on expert e (`full_probs` column mean) — Switch's `P_i`. `f_e` carries no gradient
    (it comes from a hard `topk`/`scatter` dispatch decision); `P_e` does — so gradient descent on
    this loss pushes down the router's soft probability on over-loaded experts and up on
    under-loaded ones, exactly the mechanism selftest (c) exercises. Minimized (== 0 up to the
    `alpha`/`E` scale) when both `f` and `P` are uniform at `1/n_experts`.
    """
    f_e = route.dispatch_mask.mean(dim=0)
    p_e = route.full_probs.mean(dim=0)
    return alpha * n_experts * (f_e * p_e).sum()


def training_loss(
    pred: torch.Tensor, target: torch.Tensor, route: RouterOutputs, n_experts: int, alpha: float = ALPHA, task: Task = Task.MSE
) -> tuple[torch.Tensor, dict]:
    """Total training loss `task_loss(pred, target) + L_aux` (Switch eq. 4; `L_aux` already carries `alpha`).

    `task_loss` is `MSELoss` (`pred`/`target` both `(N, 1)` float) for `Task.MSE`, or
    `CrossEntropyLoss` (`pred` `(N, n_classes)` logits, `target` `(N,)` long class indices) for
    `Task.CE` (F6 flag) — the router/aux-loss machinery is identical either way.

    Returns the scalar total (for `.backward()`) plus a plain-float breakdown for logging. No
    second `alpha` multiply here: `load_balance_aux_loss` already returns the fully-scaled Switch
    term, so `total = task_loss + aux_loss` is the complete eq.-4-plus-task-loss objective —
    `task_loss + alpha * aux_loss` would double-apply `alpha` against the frozen M1 config.
    """
    task_loss = nnf.mse_loss(pred, target) if task is Task.MSE else nnf.cross_entropy(pred, target)
    aux = load_balance_aux_loss(route, n_experts, alpha)
    total = task_loss + aux
    return total, {"task_loss": task_loss.item(), "aux_loss": aux.item(), "total_loss": total.item()}


def routing_diagnostics(route: RouterOutputs, n_experts: int) -> dict:
    """Per-expert load, router entropy, and collapsed-expert count off one `route()` call.

    `load` is `f_e` (the same dispatch-fraction column mean the aux loss uses) — the per-expert
    load histogram. `router_entropy` is the batch-mean Shannon entropy of the FULL (unmasked)
    router distribution (max `log(n_experts)` at perfect uniformity, 0 at a one-hot router),
    measured on `full_probs` rather than the sparse `gate_probs` so it reads specialization/
    collapse across every expert, not just the top-k winners. `collapsed_expert_count` counts
    experts whose load falls below `1 / (4 * n_experts)` (the M2 brief's threshold) — routing
    collapse is a first-class diagnostic here, never filtered out of the summary.
    """
    load = route.dispatch_mask.mean(dim=0)
    router_entropy = -(route.full_probs * route.full_probs.clamp_min(1e-12).log()).sum(dim=1).mean()
    collapse_threshold = 1.0 / (_COLLAPSE_LOAD_DIVISOR * n_experts)
    collapsed = int((load < collapse_threshold).sum().item())
    return {
        "load": [round(v, 6) for v in load.detach().cpu().tolist()],
        "router_entropy": router_entropy.item(),
        "collapsed_expert_count": collapsed,
        "collapse_threshold": collapse_threshold,
        "n_experts": n_experts,
    }


def moe_flops_and_params(n_experts: int, expert_hidden: int, d_in: int, output_size: int, top_k: int) -> tuple[int, int]:
    """Total params + top-`top_k`-executed MACs for one MoE shape, generalized over `output_size`.

    Same shape as `capacity_accounting.MoEShapeDescriptor`'s S2 formulas (router always scores
    every expert; only `top_k` experts' own forward passes execute), generalized from S2's
    hardcoded scalar per-expert output to an arbitrary `output_size` — see module docstring's F6
    `task={mse,ce}` section for why this lives here instead of in `capacity_accounting.py`.
    Selftest (e) proves this is byte-identical to S2's `MoEShapeDescriptor` at `output_size=1`.
    """
    router_params = d_in * n_experts + n_experts
    per_expert_params = (d_in * expert_hidden + expert_hidden) + (expert_hidden * output_size + output_size)
    total_params = router_params + n_experts * per_expert_params

    router_macs = d_in * n_experts
    per_expert_macs = d_in * expert_hidden + expert_hidden * output_size
    total_flops = router_macs + top_k * per_expert_macs
    return total_params, total_flops


def match_to_reference(
    reference: nn.Module,
    reference_config: int,
    *,
    n_experts: int = N_EXPERTS_PRIMARY,
    top_k: int = TOP_K_PRIMARY,
    d_in: int = D_IN_DEFAULT,
    output_size: int = 1,
    tol: float = MATCH_TOLERANCE,
) -> dict:
    """Sizes `expert_hidden` so an `(n_experts, top_k, d_in)` MoE matches `reference`'s params AND FLOPs.

    `reference` is a real FlexNN/width-net module; `reference_config` is its routed capacity
    (depth for FlexNN, width for a width net). Both accounting quantities come from
    `capacity_accounting.param_count`/`executed_flops` — the S2 module that is the ONLY source of
    params/FLOPs numbers in this programme (`docs/plans/capacity_programme/shared/metrics-accounting.md`).
    `param_count` excludes any `logvar`-named parameter (`capacity_accounting.LOGVAR_HEAD_PATH_SUBSTRING`):
    MASTER Decision 2 (MSE-only) makes a width net's logvar head dead weight, and this MoE's
    experts have no logvar branch to compare it against.

    `expert_hidden` is this net's ONE free sizing knob; the MoE side's own (params, FLOPs) are
    EXACTLY affine in `expert_hidden` for fixed `(n_experts, top_k, d_in, output_size)`, so both
    targets are solved in closed form (two-point slope from `h=0`/`h=1`) and then refined over a
    small window of nearby integers, picking the `expert_hidden` that minimizes the WORSE of the
    two relative errors. A single knob cannot generally zero both errors at once (params and
    FLOPs trace one straight line in (params, FLOPs) space as `expert_hidden` varies), so a good
    joint match exists only when the reference's own (params, FLOPs) pair sits near that line —
    this function reports the achieved errors either way, it does not hide a bad match.

    `output_size=1` (the M2-frozen MSE shape) computes the MoE side via
    `capacity_accounting.MoEShapeDescriptor` unchanged — S2 remains the only source for the shape
    it supports. `output_size>1` (F6's `Task.CE`) falls back to `moe_flops_and_params` (this
    module), since S2's descriptor hardcodes a scalar per-expert output — see module docstring.

    Returns:
        A dict with the target/achieved params and FLOPs, both relative errors, the chosen
        `expert_hidden`, `within_tolerance` (both errors `<= tol`), and the exact-solve
        intermediates (`h_from_params_only`/`h_from_flops_only`) — the complete matching
        arithmetic, meant to be dumped verbatim into a summary JSON.
    """
    target_params = ca.param_count(reference, path_filter=ca.LOGVAR_HEAD_PATH_SUBSTRING)
    target_flops = ca.executed_flops(reference, reference_config)

    def params_at(h: int) -> int:
        if output_size == 1:
            return ca.param_count(ca.MoEShapeDescriptor(d_in=d_in, expert_hidden=h, n_experts=n_experts))
        return moe_flops_and_params(n_experts, h, d_in, output_size, top_k)[0]

    def flops_at(h: int) -> int:
        if output_size == 1:
            return ca.executed_flops(ca.MoEShapeDescriptor(d_in=d_in, expert_hidden=h, n_experts=n_experts), top_k)
        return moe_flops_and_params(n_experts, h, d_in, output_size, top_k)[1]

    p0, p1 = params_at(0), params_at(1)
    f0, f1 = flops_at(0), flops_at(1)
    h_from_params = (target_params - p0) / (p1 - p0) if p1 != p0 else None
    h_from_flops = (target_flops - f0) / (f1 - f0) if f1 != f0 else None

    candidates = {1}
    for h_real in (h_from_params, h_from_flops):
        if h_real is not None:
            center = max(1, round(h_real))
            candidates.update(range(max(1, center - _MATCH_REFINE_WINDOW), center + _MATCH_REFINE_WINDOW + 1))

    best_h, best_params, best_flops, best_score = None, None, None, None
    for h in sorted(candidates):
        p, f = params_at(h), flops_at(h)
        score = max(abs(p - target_params) / target_params, abs(f - target_flops) / target_flops)
        if best_score is None or score < best_score:
            best_h, best_params, best_flops, best_score = h, p, f, score

    params_rel_err = abs(best_params - target_params) / target_params
    flops_rel_err = abs(best_flops - target_flops) / target_flops
    return {
        "reference_class": type(reference).__name__,
        "reference_config": reference_config,
        "n_experts": n_experts,
        "top_k": top_k,
        "d_in": d_in,
        "output_size": output_size,
        "target_params": target_params,
        "target_flops": target_flops,
        "expert_hidden": best_h,
        "achieved_params": best_params,
        "achieved_flops": best_flops,
        "params_rel_err": params_rel_err,
        "flops_rel_err": flops_rel_err,
        "tol": tol,
        "within_tolerance": params_rel_err <= tol and flops_rel_err <= tol,
        "h_from_params_only": h_from_params,
        "h_from_flops_only": h_from_flops,
    }


# ---------------------------------------------------------------------------
# Selftest -- no training batteries (that is M3); (c) runs a small router-only optimization, the
# rest are random-init/no-training checks. MUST pass before any real read.
# ---------------------------------------------------------------------------


def _assert_shapes_finite(net: MoERegressionNet, x: torch.Tensor, expected_output_size: int = 1) -> tuple[bool, str]:
    """(a) Both routing modes return finite `(N, expected_output_size)` predictions."""
    ok = True
    detail = []
    for mode in RoutingMode:
        y = net(x, mode=mode)
        shape_ok = tuple(y.shape) == (x.shape[0], expected_output_size)
        finite_ok = bool(torch.isfinite(y).all())
        ok = ok and shape_ok and finite_ok
        detail.append(f"{mode.value}: shape_ok={shape_ok} finite_ok={finite_ok}")
    return ok, "; ".join(detail)


def _assert_topk_gradient_isolation(net: MoERegressionNet, x: torch.Tensor) -> tuple[bool, float]:
    """(b) Perturbing a non-selected expert's weights leaves every non-selecting row's output unchanged."""
    with torch.no_grad():
        route0 = net.route(x)
        y0 = net(x, mode=RoutingMode.TOP_K)
    max_err = 0.0
    ok_all = True
    for e_idx, expert in enumerate(net.experts):
        not_selected = route0.dispatch_mask[:, e_idx] == 0.0
        if not bool(not_selected.any()):
            continue
        originals = [p.detach().clone() for p in expert.parameters()]
        with torch.no_grad():
            for p in expert.parameters():
                p.add_(torch.randn_like(p) * 5.0)
            y1 = net(x, mode=RoutingMode.TOP_K)
            for p, orig in zip(expert.parameters(), originals, strict=True):
                p.copy_(orig)
        err = (y0[not_selected] - y1[not_selected]).abs().max().item()
        max_err = max(max_err, err)
        ok_all = ok_all and (err < _TOPK_ISOLATION_TOL)
    return ok_all, max_err


def _assert_aux_loss_decreases(net: MoERegressionNet, x: torch.Tensor) -> tuple[bool, float, float]:
    """(c) Deliberately imbalances the router, then confirms gradient descent on `L_aux` alone reduces it."""
    with torch.no_grad():
        net.router.weight.zero_()
        net.router.bias.zero_()
        net.router.bias[0] = 10.0  # forces expert 0 into every row's top-k set at init
    aux0 = load_balance_aux_loss(net.route(x), net.n_experts).item()
    opt = torch.optim.Adam(net.router.parameters(), lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        aux = load_balance_aux_loss(net.route(x), net.n_experts)
        aux.backward()
        opt.step()
    aux_final = load_balance_aux_loss(net.route(x), net.n_experts).item()
    return aux_final < aux0, aux0, aux_final


def _assert_matching_helper() -> tuple[bool, list[dict]]:
    """(d) `match_to_reference` hits <=5% on two reference configs from two different families."""
    flex = FlexibleHiddenLayersNN(
        input_size=D_IN_DEFAULT, hidden_size=32, output_size=1, max_hidden_layers=7, layer_selection_method=LayerSelectionMethod.NONE, n_predictor_layers=0
    )
    flex.build_model()
    width_net = nwn.NestedWidthNet(w_max=256)

    results = [
        match_to_reference(flex.model, 2),  # routed depth 2 of 7 stored blocks
        match_to_reference(width_net, 64),  # routed width 64 of 256
    ]
    ok_all = all(r["within_tolerance"] for r in results)
    return ok_all, results


def _assert_moe_flops_and_params_generalization() -> tuple[bool, dict]:
    """(e) F6: agreement + hand-check for the generalized MoE accounting.

    `moe_flops_and_params` agrees with S2's `MoEShapeDescriptor` at output_size=1, and a
    hand-computed output_size=5 (CE-shaped) config checks out against a manual derivation.
    """
    n_experts, expert_hidden, d_in, top_k = 8, 6, 3, 2
    s2_desc = ca.MoEShapeDescriptor(d_in=d_in, expert_hidden=expert_hidden, n_experts=n_experts)
    s2_params = ca.param_count(s2_desc)
    s2_flops = ca.executed_flops(s2_desc, top_k)
    gen_params_1, gen_flops_1 = moe_flops_and_params(n_experts, expert_hidden, d_in, 1, top_k)
    agree_ok = gen_params_1 == s2_params and gen_flops_1 == s2_flops

    # Hand-derived output_size=5 (e.g. CE with 5 classes): router (3,8): 3*8+8=32.
    # per-expert: trunk (3,6): 3*6+6=24, head (6,5): 6*5+5=35 -> 59 params; 8 experts -> 472.
    expected_ce_params = 32 + 8 * 59
    # FLOPs: router 3*8=24 (always full) + top_k=2 * [trunk (3,6)=18 + head (6,5)=30] = 24 + 2*48 = 120.
    expected_ce_flops = 24 + 2 * 48
    gen_params_5, gen_flops_5 = moe_flops_and_params(n_experts, expert_hidden, d_in, 5, top_k)
    ce_ok = gen_params_5 == expected_ce_params and gen_flops_5 == expected_ce_flops

    ok = agree_ok and ce_ok
    detail = {
        "s2_at_output_1": (s2_params, s2_flops),
        "generalized_at_output_1": (gen_params_1, gen_flops_1),
        "agree_at_output_1": agree_ok,
        "generalized_at_output_5": (gen_params_5, gen_flops_5),
        "expected_at_output_5": (expected_ce_params, expected_ce_flops),
        "hand_computed_output_5_ok": ce_ok,
    }
    return ok, detail


def _assert_ce_task_trains(n_classes: int = 5, d_in: int = 3) -> tuple[bool, float, float]:
    """(f) F6: `Task.CE` shape/isolation/training checks.

    A `Task.CE` net's shapes/gradient-isolation hold, and `training_loss(task=Task.CE)` genuinely
    reduces cross-entropy + aux loss under gradient descent on synthetic class data.
    """
    torch.manual_seed(0)
    net = MoERegressionNet(n_experts=N_EXPERTS_PRIMARY, top_k=TOP_K_PRIMARY, expert_hidden=EXPERT_HIDDEN_DEFAULT, d_in=d_in, output_size=n_classes)
    x = torch.randn(64, d_in)
    y = torch.randint(0, n_classes, (64,))

    net.eval()
    ok_shapes, _ = _assert_shapes_finite(net, x, expected_output_size=n_classes)
    ok_iso, _ = _assert_topk_gradient_isolation(net, x)
    if not (ok_shapes and ok_iso):
        return False, float("nan"), float("nan")

    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    route0 = net.route(x)
    pred0 = net(x, mode=RoutingMode.TOP_K)
    loss0, _ = training_loss(pred0, y, route0, net.n_experts, task=Task.CE)
    for _ in range(100):
        opt.zero_grad()
        route = net.route(x)
        pred = net(x, mode=RoutingMode.TOP_K)
        loss, _ = training_loss(pred, y, route, net.n_experts, task=Task.CE)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        route_final = net.route(x)
        pred_final = net(x, mode=RoutingMode.TOP_K)
        loss_final, _ = training_loss(pred_final, y, route_final, net.n_experts, task=Task.CE)
    return loss_final.item() < loss0.item(), loss0.item(), loss_final.item()


def run_selftest() -> bool:
    """Runs all selftest checks on a fixed seed and prints PASS/FAIL for each."""
    torch.manual_seed(0)
    net = MoERegressionNet(n_experts=N_EXPERTS_PRIMARY, top_k=TOP_K_PRIMARY, expert_hidden=EXPERT_HIDDEN_DEFAULT, d_in=D_IN_DEFAULT)
    net.eval()
    x = torch.randn(37, D_IN_DEFAULT)

    ok_shapes, detail = _assert_shapes_finite(net, x)
    print(f"[moe_regression selftest] (a) shapes/finite: {detail}  {'PASS' if ok_shapes else 'FAIL'}")

    ok_iso, err_iso = _assert_topk_gradient_isolation(net, x)
    print(f"[moe_regression selftest] (b) top-k gradient isolation: max_abs_err={err_iso:.3e} (tol={_TOPK_ISOLATION_TOL:.0e})  {'PASS' if ok_iso else 'FAIL'}")

    net_aux = MoERegressionNet(n_experts=N_EXPERTS_PRIMARY, top_k=TOP_K_PRIMARY, expert_hidden=EXPERT_HIDDEN_DEFAULT, d_in=D_IN_DEFAULT)
    x_aux = torch.randn(64, D_IN_DEFAULT)
    ok_aux, aux0, aux1 = _assert_aux_loss_decreases(net_aux, x_aux)
    print(f"[moe_regression selftest] (c) aux-loss decreases under imbalance: {aux0:.4f} -> {aux1:.4f}  {'PASS' if ok_aux else 'FAIL'}")

    ok_match, match_results = _assert_matching_helper()
    for r in match_results:
        print(
            f"[moe_regression selftest] (d) match_to_reference({r['reference_class']}, config={r['reference_config']}): "
            f"expert_hidden={r['expert_hidden']} target=({r['target_params']}, {r['target_flops']}) "
            f"achieved=({r['achieved_params']}, {r['achieved_flops']}) "
            f"params_rel_err={r['params_rel_err']:.4f} flops_rel_err={r['flops_rel_err']:.4f} within_tolerance={r['within_tolerance']}"
        )
    print(f"[moe_regression selftest] (d) matching helper: {'PASS' if ok_match else 'FAIL'}")

    ok_gen, gen_detail = _assert_moe_flops_and_params_generalization()
    print(
        f"[moe_regression selftest] (e) moe_flops_and_params generalization: "
        f"s2_at_1={gen_detail['s2_at_output_1']} generalized_at_1={gen_detail['generalized_at_output_1']} "
        f"generalized_at_5={gen_detail['generalized_at_output_5']} expected_at_5={gen_detail['expected_at_output_5']}  "
        f"{'PASS' if ok_gen else 'FAIL'}"
    )

    ok_ce, ce_loss0, ce_loss1 = _assert_ce_task_trains()
    print(f"[moe_regression selftest] (f) Task.CE shapes/isolation + training_loss decreases: {ce_loss0:.4f} -> {ce_loss1:.4f}  {'PASS' if ok_ce else 'FAIL'}")

    ok = ok_shapes and ok_iso and ok_aux and ok_match and ok_gen and ok_ce
    print(f"[moe_regression selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Parses args and runs the selftest, or prints help (this module has no standalone real-run mode)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="No-training / small-optimization known-answer checks.")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
