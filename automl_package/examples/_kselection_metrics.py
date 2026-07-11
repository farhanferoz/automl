"""Shared k-selection diagnostics for ProbReg sweeps.

Computes a richer set of metrics than `effective_k` alone so that we can
distinguish "uniform spread", "midpoint-peaked", and "saturated at k_max"
distributions — all of which give the same E[k] under the existing metric.

Diagnostic interpretations at k_max=K (with n_k = K-1 non-bypass modes,
k ∈ {2..K}):

    selection_perplexity = exp(H(mean_p_nobypass))
        Uniform on n_k modes → n_k.
        Delta on one mode    → 1.
        Distinguishes "uniform" from "concentrated", which E[k] alone cannot.

    dead_mode_count
        Number of modes whose batch-mean probability < 0.5 / n_k.
        Detects MoE-style starvation.

    mean_max_p
        Mean over batch of per-sample max selection probability.
        Proxy for classifier confidence (CE-bottleneck behaviour).

    marginal_p (numpy array, shape (n_k,))
        Per-mode batch-mean probability over non-bypass modes.
        For inspecting distribution shape directly.

`effective_k`, `effective_k_nobypass`, and `bypass_fraction` retain their
historical meaning for backward compatibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def compute_kselection_metrics(model: Any, x_te: np.ndarray, k_max: int) -> dict[str, Any]:
    """Compute k-selection metrics from the n_classes_predictor over x_te.

    Args:
        model: A fitted ProbabilisticRegressionModel.
        x_te: Test features, shape (N, D).
        k_max: Maximum k value (top of the {2..k_max} range).

    Returns:
        Dict with scalar metrics plus `marginal_p` numpy array. If the model
        has no n_classes_predictor (static k), returns degenerate values
        appropriate for "fixed k=k_max".
    """
    degenerate = {
        "effective_k": float(k_max),
        "effective_k_nobypass": float(k_max),
        "bypass_fraction": 0.0,
        "selection_perplexity": float("nan"),
        "dead_mode_count": 0,
        "mean_max_p": float("nan"),
        "marginal_p": np.array([], dtype=np.float32),
    }
    if not hasattr(model, "model") or not hasattr(model.model, "n_classes_predictor"):
        return degenerate
    if model.model.n_classes_predictor is None:
        return degenerate

    x_t = torch.tensor(x_te, dtype=torch.float32).to(model.device)
    model.model.eval()
    with torch.no_grad():
        logits = model.model.n_classes_predictor(x_t)
        probs = F.softmax(logits, dim=-1)  # (N, n_modes); last column = bypass
        n_modes = probs.size(1)
        n_k = n_modes - 1  # non-bypass modes correspond to k ∈ {2..k_max}
        k_vals_full = torch.tensor(
            [i + 2 for i in range(n_k)] + [k_max],
            dtype=torch.float32,
        ).to(model.device)

        effective_k_per_sample = (probs * k_vals_full).sum(-1)
        effective_k = float(effective_k_per_sample.mean().cpu().item())

        bypass_mass = probs[:, -1]
        bypass_fraction = float(bypass_mass.mean().cpu().item())

        prob_probs = probs[:, :-1]  # non-bypass slice
        denom = prob_probs.sum(-1, keepdim=True).clamp_min(1e-8)
        renorm = prob_probs / denom  # per-sample, conditional on non-bypass
        prob_k_vals = k_vals_full[:-1]

        expected_k_prob = (renorm * prob_k_vals).sum(-1)
        mask = (1.0 - bypass_mass) > 1e-6
        if mask.any():
            effective_k_nobypass = float(expected_k_prob[mask].mean().cpu().item())
        else:
            effective_k_nobypass = float("nan")

        marginal_p = renorm.mean(dim=0)  # batch-marginal over non-bypass modes
        marginal_p_np = marginal_p.cpu().numpy().astype(np.float32)

        eps = 1e-12
        entropy = -(marginal_p * (marginal_p.clamp_min(eps).log())).sum()
        selection_perplexity = float(torch.exp(entropy).cpu().item())

        dead_threshold = 0.5 / max(n_k, 1)
        dead_mode_count = int((marginal_p < dead_threshold).sum().cpu().item())

        max_per_sample = renorm.max(dim=-1).values
        mean_max_p = float(max_per_sample.mean().cpu().item())

    return {
        "effective_k": effective_k,
        "effective_k_nobypass": effective_k_nobypass,
        "bypass_fraction": bypass_fraction,
        "selection_perplexity": selection_perplexity,
        "dead_mode_count": dead_mode_count,
        "mean_max_p": mean_max_p,
        "marginal_p": marginal_p_np,
    }


def weight_survival_metrics(weights: np.ndarray, floor_factor: float = 0.5) -> dict[str, Any]:
    """Surviving-k diagnostics for the variational-EM k-selector.

    Operates on the inferred mixing weights ``w̄`` (the Dirichlet posterior mean
    ``γ_c / Σ_j γ_j`` for Basis A, or the per-input mean averaged over inputs for
    Basis B) — NOT on the as-built selector. This is the "what to record" of the
    controlled test in docs/kselection_variational_em_2026-06-13 (note §9).

    Args:
        weights: 1-D array of mixing weights over the K classes (sums to ~1).
            By convention the last entry is the bypass class.
        floor_factor: a class counts as surviving if its weight exceeds
            ``floor_factor / K``. The note uses ``1/(2K)`` (floor_factor=0.5):
            half the weight a class would carry if all K shared equally.

    Returns:
        dict with:
          - surviving_k: number of weights above the floor (the headline integer).
          - effective_number: exp(H), H = -Σ_c w̄_c ln w̄_c (continuous cross-check;
            1 when all weight on one class, K when spread evenly).
          - bypass_weight: weight on the last (bypass) class.
          - max_weight: largest single weight.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    w = w / max(w.sum(), 1e-12)
    k_total = w.size
    floor = floor_factor / max(k_total, 1)
    surviving_k = int((w > floor).sum())
    eps = 1e-12
    entropy = float(-(w * np.log(np.clip(w, eps, None))).sum())
    return {
        "surviving_k": surviving_k,
        "effective_number": float(np.exp(entropy)),
        "bypass_weight": float(w[-1]),
        "max_weight": float(w.max()),
    }
