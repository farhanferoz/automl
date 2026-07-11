"""Variational-EM k-selector (Basis A): one-model harness for the controlled toy.

Implements the closed-form variational EM of
docs/kselection_variational_em_2026-06-13/kselection_variational_em.{md,pdf}:

  E-step (closed form, needs the targets, detached):
      r_ic ∝ φ_ic · exp(ψ(γ_c)),     φ_ic = N(y_i; μ_c(x_i), σ_c²(x_i))
  network gradient step (classifier + heads, ONE model):
      maximise   Σ_i Σ_c r_ic ln φ_ic
  M-step (closed form, global Dirichlet concentrations):
      γ_c = α₀ + Σ_i r_ic

The model is one network: a classifier (x → bin logits) feeding per-bin regression
heads, plus a bypass head (x → mean, log-variance) as the K-th class. Mixing weights
are GLOBAL (the posterior mean w̄ = γ/Σγ) — this is the grounding (Basis A) case used
for the Step-1 toy; the per-input Basis B is future work. The number of classes ``k`` is
read off how many weights survive the sparse Dirichlet prior, with NO separate k-selector.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_log_density(y: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Per-class Gaussian log-density ``ln N(y; mu, exp(log_var))``.

    Args:
        y: ``(N,)`` or ``(N, 1)`` targets.
        mu: ``(N, K)`` per-class means.
        log_var: ``(N, K)`` per-class log-variances.

    Returns:
        ``(N, K)`` log-densities.
    """
    y = y.view(-1, 1)
    return -0.5 * (math.log(2.0 * math.pi) + log_var + (y - mu) ** 2 * torch.exp(-log_var))


class _MLP(nn.Module):
    """Small ReLU MLP."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 32, layers: int = 2) -> None:
        """Builds an MLP with ``layers`` hidden blocks of width ``hidden``."""
        super().__init__()
        mods: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class VariationalEMKSelector(nn.Module):
    """One-model variational-EM k-selector with global (Basis A) weights.

    The number of discretisation classes (bins) is ``k_max``; the bypass is the
    ``K = k_max + 1``-th class. Global Dirichlet concentrations ``gamma`` are
    updated by the closed-form M-step and define the mixing weights.
    """

    def __init__(
        self,
        input_dim: int,
        k_max: int,
        alpha0: float = 0.1,
        bin_centroids: np.ndarray | None = None,
        marginal_mean: float = 0.0,
        marginal_log_var: float = 0.0,
        hidden: int = 32,
        log_var_min: float = -8.0,
        log_var_max: float = 4.0,
    ) -> None:
        """Initializes the model and the global concentrations buffer.

        Args:
            input_dim: feature dimension.
            k_max: number of discretisation classes (bins); total classes ``K = k_max + 1``.
            alpha0: symmetric Dirichlet prior concentration (use < 1 to prune).
            bin_centroids: optional length-``k_max`` array; each bin head's mean is
                initialised here so the bins tile the target range.
            marginal_mean: initial bypass mean (set to the data mean for a fair start).
            marginal_log_var: initial bypass log-variance.
            hidden: hidden width of the sub-networks.
            log_var_min: lower clamp on predicted log-variances (stability).
            log_var_max: upper clamp on predicted log-variances (stability).
        """
        super().__init__()
        self.k_max = k_max
        self.K = k_max + 1
        self.alpha0 = float(alpha0)
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        self.classifier = _MLP(input_dim, k_max, hidden)
        self.bin_heads = nn.ModuleList([_MLP(1, 2, hidden) for _ in range(k_max)])
        self.bypass_head = _MLP(input_dim, 2, hidden)

        if bin_centroids is not None:
            for c, head in enumerate(self.bin_heads):
                with torch.no_grad():
                    head.net[-1].bias[0] = float(bin_centroids[c])
                    head.net[-1].bias[1] = 0.0
        with torch.no_grad():
            self.bypass_head.net[-1].bias[0] = float(marginal_mean)
            self.bypass_head.net[-1].bias[1] = float(marginal_log_var)

        self.register_buffer("gamma", torch.full((self.K,), self.alpha0))

    def _clamp_lv(self, lv: torch.Tensor) -> torch.Tensor:
        """Clamps a log-variance for numerical stability."""
        return lv.clamp(self.log_var_min, self.log_var_max)

    def per_class_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns per-class ``(mu, log_var)``, each ``(N, K)`` with the bypass last."""
        pi = F.softmax(self.classifier(x), dim=-1)
        mus, lvs = [], []
        for c in range(self.k_max):
            out = self.bin_heads[c](pi[:, c : c + 1])
            mus.append(out[:, 0])
            lvs.append(self._clamp_lv(out[:, 1]))
        byp = self.bypass_head(x)
        mus.append(byp[:, 0])
        lvs.append(self._clamp_lv(byp[:, 1]))
        return torch.stack(mus, dim=1), torch.stack(lvs, dim=1)

    def responsibilities(self, y: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form E-step ``r_ic ∝ φ_ic exp(ψ(γ_c))``.

        Returns:
            ``(r, log_phi)`` where ``r`` is detached (a fixed target for the fit step).
        """
        log_phi = gaussian_log_density(y, mu, log_var)
        e_log_w = torch.digamma(self.gamma) - torch.digamma(self.gamma.sum())
        r = F.softmax(log_phi + e_log_w.unsqueeze(0), dim=-1)
        return r.detach(), log_phi

    def fit_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative responsibility-weighted fit ``-mean_i Σ_c r_ic ln φ_ic`` (network objective)."""
        mu, log_var = self.per_class_params(x)
        r, log_phi = self.responsibilities(y, mu, log_var)
        return -(r * log_phi).sum(dim=1).mean()

    @torch.no_grad()
    def m_step(self, x_all: torch.Tensor, y_all: torch.Tensor) -> None:
        """Closed-form M-step ``γ_c = α₀ + Σ_i r_ic`` over the full data."""
        mu, log_var = self.per_class_params(x_all)
        r, _ = self.responsibilities(y_all, mu, log_var)
        self.gamma = self.alpha0 + r.sum(dim=0)

    @torch.no_grad()
    def mean_weights(self) -> np.ndarray:
        """Posterior-mean mixing weights ``w̄ = γ / Σγ`` as a numpy array."""
        return (self.gamma / self.gamma.sum()).cpu().numpy()

    @torch.no_grad()
    def mixture_nll(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Mean NLL under the genuine mixture with posterior-mean weights."""
        mu, log_var = self.per_class_params(x)
        log_phi = gaussian_log_density(y, mu, log_var)
        log_w = torch.log((self.gamma / self.gamma.sum()).clamp_min(1e-12))
        return float((-torch.logsumexp(log_w.unsqueeze(0) + log_phi, dim=-1)).mean().item())


def train_variational_em(
    x: np.ndarray,
    y: np.ndarray,
    k_max: int,
    alpha0: float = 0.1,
    n_epochs: int = 400,
    lr: float = 1e-2,
    m_step_every: int = 1,
    hidden: int = 32,
    device: str = "cpu",
    seed: int = 0,
) -> VariationalEMKSelector:
    """Fits a :class:`VariationalEMKSelector` by alternating gradient and M-steps.

    Bin-head means are initialised at evenly spaced quantiles of ``y`` (so the bins
    tile the target range) and the bypass at the marginal Gaussian. Training then
    alternates a full-batch gradient step on the responsibility-weighted fit with
    the closed-form M-step on the global concentrations.

    Args:
        x: features, shape ``(N, D)`` or ``(N,)``.
        y: targets, shape ``(N,)``.
        k_max: number of discretisation classes (bins).
        alpha0: Dirichlet prior concentration.
        n_epochs: number of (gradient + optional M-step) iterations.
        lr: Adam learning rate.
        m_step_every: run the M-step every this many epochs.
        hidden: hidden width.
        device: torch device string.
        seed: RNG seed.

    Returns:
        The fitted model.
    """
    torch.manual_seed(seed)
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=np.float32).ravel()
    x_t = torch.as_tensor(x_arr, device=device)
    y_t = torch.as_tensor(y_arr, device=device)

    quantiles = np.linspace(0.0, 100.0, k_max + 2)[1:-1]
    centroids = np.percentile(y_arr, quantiles)

    model = VariationalEMKSelector(
        input_dim=x_t.shape[1],
        k_max=k_max,
        alpha0=alpha0,
        bin_centroids=centroids,
        marginal_mean=float(y_arr.mean()),
        marginal_log_var=float(np.log(max(y_arr.var(), 1e-4))),
        hidden=hidden,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.m_step(x_t, y_t)  # seed γ from the centroid-tiled initialisation
    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = model.fit_loss(x_t, y_t)
        loss.backward()
        opt.step()
        if (epoch + 1) % m_step_every == 0:
            model.m_step(x_t, y_t)
    return model
