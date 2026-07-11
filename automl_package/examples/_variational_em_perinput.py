"""Per-input variational-EM k-selector (Basis B): mixing weights are a function of x.

Basis A (``_variational_em.py``) used GLOBAL mixing weights with a closed-form Dirichlet
M-step ``γ = α₀ + Σ_i r_ic`` — one "how many buckets" answer for the whole dataset. Basis B
makes the weights per-input: a ``weight_net`` maps ``x`` to Dirichlet concentrations
``γ(x)``, and the posterior-mean weights ``w(x) = γ(x)/Σγ(x)`` mix the per-class densities.
There is no closed form for amortised per-input concentrations, so the M-step is replaced by
gradient descent on a proper per-input ELBO:

    maximise   mean_i [ logsumexp_c( ln φ_ic + E_q[ln w_ic] )  −  KL(Dir(γ_i) ‖ Dir(α₀)) ]
    with       E_q[ln w_ic] = ψ(γ_ic) − ψ(Σ_c γ_ic)

The KL is the model's own term (coefficient 1, no tuned λ): with ``α₀ < 1`` the Dirichlet
prior favours putting each input's weight on FEW classes, so "how many buckets at x" is read
off how many of ``w(x)`` survive — per input, with NO separate k-selector. The bin/bypass
heads now read ``x`` directly (Basis A fed the bin head its own class probability, which only
moves the mean through one scalar — too weak when the modes move with ``x``).

Whether this per-input objective actually prunes (vs collapsing to uniform) is the open Basis
B question; this module is the instrument to answer it, not a guarantee.
"""

from __future__ import annotations

import _variational_em as vem  # _MLP, gaussian_log_density
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def dirichlet_kl(gamma: torch.Tensor, alpha0: float) -> torch.Tensor:
    """KL divergence ``KL(Dir(γ) ‖ Dir(α₀·1))`` for batched concentrations.

    Args:
        gamma: ``(N, K)`` positive concentrations.
        alpha0: symmetric prior concentration.

    Returns:
        ``(N,)`` per-input KL (``≥ 0``, zero iff ``γ = α₀·1``).
    """
    a0 = torch.full_like(gamma, alpha0)
    sg = gamma.sum(-1)
    sa = a0.sum(-1)
    base = torch.lgamma(sg) - torch.lgamma(gamma).sum(-1) - torch.lgamma(sa) + torch.lgamma(a0).sum(-1)
    return base + ((gamma - a0) * (torch.digamma(gamma) - torch.digamma(sg).unsqueeze(-1))).sum(-1)


class PerInputVariationalEMKSelector(nn.Module):
    """One-model variational-EM k-selector with PER-INPUT Dirichlet weights (Basis B).

    The bypass is the ``K = k_max + 1``-th class. ``concentrations(x)`` gives the per-input
    Dirichlet ``γ(x)``; ``weights(x)`` its posterior mean. Trained by gradient on
    :meth:`elbo_loss` — no closed-form M-step.
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
        gamma_floor: float = 1e-2,
        adaptive_bin_means: bool = True,
    ) -> None:
        """Initializes the weight net and the per-class heads.

        Args:
            input_dim: feature dimension.
            k_max: number of discretisation classes (bins); total ``K = k_max + 1``.
            alpha0: symmetric Dirichlet prior concentration (``< 1`` to favour per-input sparsity).
            bin_centroids: optional length-``k_max`` array; seeds (``adaptive_bin_means=True``) or
                FIXES (``adaptive_bin_means=False``) each bin's mean so the bins tile the target range.
            marginal_mean: initial bypass mean (the data mean for a fair start).
            marginal_log_var: initial bypass log-variance.
            hidden: hidden width of the sub-networks.
            log_var_min: lower clamp on predicted log-variances.
            log_var_max: upper clamp on predicted log-variances.
            gamma_floor: added to ``softplus`` so concentrations stay strictly positive.
            adaptive_bin_means: if True, each bin is an ``x``-dependent head (a bin can slide its
                Gaussian to fit, so the weights need not vary with ``x`` — they collapse to global).
                If False, bin MEANS are fixed at ``bin_centroids`` with per-bin learnable widths, so
                moving probability mass as ``x`` changes MUST go through the per-input weights — the
                faithful "classifier over fixed bins" design.
        """
        super().__init__()
        self.k_max = k_max
        self.K = k_max + 1
        self.alpha0 = float(alpha0)
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        self.gamma_floor = float(gamma_floor)
        self.adaptive_bin_means = adaptive_bin_means

        self.weight_net = vem._MLP(input_dim, self.K, hidden)
        self.bypass_head = vem._MLP(input_dim, 2, hidden)

        if adaptive_bin_means:
            self.bin_heads = nn.ModuleList([vem._MLP(input_dim, 2, hidden) for _ in range(k_max)])
            if bin_centroids is not None:
                for c, head in enumerate(self.bin_heads):
                    with torch.no_grad():
                        head.net[-1].bias[0] = float(bin_centroids[c])
                        head.net[-1].bias[1] = 0.0
        else:
            centers = np.zeros(k_max, dtype=np.float32) if bin_centroids is None else np.asarray(bin_centroids, dtype=np.float32)
            self.register_buffer("bin_means", torch.as_tensor(centers))
            self.bin_log_vars = nn.Parameter(torch.full((k_max,), float(marginal_log_var)))
        with torch.no_grad():
            self.bypass_head.net[-1].bias[0] = float(marginal_mean)
            self.bypass_head.net[-1].bias[1] = float(marginal_log_var)

    def _clamp_lv(self, lv: torch.Tensor) -> torch.Tensor:
        """Clamps a log-variance for numerical stability."""
        return lv.clamp(self.log_var_min, self.log_var_max)

    def concentrations(self, x: torch.Tensor) -> torch.Tensor:
        """Per-input Dirichlet concentrations ``γ(x)``, shape ``(N, K)``, strictly positive."""
        return F.softplus(self.weight_net(x)) + self.gamma_floor

    def per_class_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns per-class ``(mu, log_var)``, each ``(N, K)`` with the bypass last."""
        n = x.shape[0]
        if self.adaptive_bin_means:
            mus, lvs = [], []
            for c in range(self.k_max):
                out = self.bin_heads[c](x)
                mus.append(out[:, 0])
                lvs.append(self._clamp_lv(out[:, 1]))
            bin_mu = torch.stack(mus, dim=1)
            bin_lv = torch.stack(lvs, dim=1)
        else:
            bin_mu = self.bin_means.unsqueeze(0).expand(n, -1)
            bin_lv = self._clamp_lv(self.bin_log_vars).unsqueeze(0).expand(n, -1)
        byp = self.bypass_head(x)
        mu = torch.cat([bin_mu, byp[:, 0:1]], dim=1)
        log_var = torch.cat([bin_lv, self._clamp_lv(byp[:, 1:2])], dim=1)
        return mu, log_var

    def elbo_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative per-input ELBO ``-mean_i[ LSE_c(ln φ_ic + E_q[ln w_ic]) - KL_i ]``."""
        gamma = self.concentrations(x)
        mu, log_var = self.per_class_params(x)
        log_phi = vem.gaussian_log_density(y, mu, log_var)
        e_log_w = torch.digamma(gamma) - torch.digamma(gamma.sum(-1, keepdim=True))
        recon = torch.logsumexp(log_phi + e_log_w, dim=-1)
        kl = dirichlet_kl(gamma, self.alpha0)
        return -(recon - kl).mean()

    @torch.no_grad()
    def weights(self, x: torch.Tensor) -> torch.Tensor:
        """Posterior-mean per-input mixing weights ``w(x) = γ(x)/Σγ(x)``, shape ``(N, K)``."""
        g = self.concentrations(x)
        return g / g.sum(-1, keepdim=True)

    @torch.no_grad()
    def mixture_nll_per_point(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Per-point NLL under the per-input posterior-mean mixture, shape ``(N,)``."""
        mu, log_var = self.per_class_params(x)
        log_phi = vem.gaussian_log_density(y, mu, log_var)
        log_w = torch.log(self.weights(x).clamp_min(1e-12))
        return -torch.logsumexp(log_w + log_phi, dim=-1)


def train_perinput_variational_em(
    x: np.ndarray,
    y: np.ndarray,
    k_max: int,
    alpha0: float = 0.1,
    n_epochs: int = 600,
    lr: float = 1e-2,
    hidden: int = 32,
    gamma_floor: float = 1e-2,
    adaptive_bin_means: bool = True,
    device: str = "cpu",
    seed: int = 0,
) -> PerInputVariationalEMKSelector:
    """Fits a :class:`PerInputVariationalEMKSelector` by gradient on the per-input ELBO.

    Bin-head means are seeded at evenly spaced quantiles of ``y`` and the bypass at the
    marginal Gaussian, matching the Basis A initialisation. There is NO M-step.

    Args:
        x: features, shape ``(N, D)`` or ``(N,)``.
        y: targets, shape ``(N,)``.
        k_max: number of discretisation classes (bins).
        alpha0: Dirichlet prior concentration.
        n_epochs: gradient iterations.
        lr: Adam learning rate.
        hidden: hidden width.
        gamma_floor: positivity floor on concentrations.
        adaptive_bin_means: see :class:`PerInputVariationalEMKSelector` (default adaptive heads).
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

    model = PerInputVariationalEMKSelector(
        input_dim=x_t.shape[1],
        k_max=k_max,
        alpha0=alpha0,
        bin_centroids=centroids,
        marginal_mean=float(y_arr.mean()),
        marginal_log_var=float(np.log(max(y_arr.var(), 1e-4))),
        hidden=hidden,
        gamma_floor=gamma_floor,
        adaptive_bin_means=adaptive_bin_means,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = model.elbo_loss(x_t, y_t)
        loss.backward()
        opt.step()
    return model


class AggregateSparsityKSelector(nn.Module):
    """Per-input softmax weights with ONE sparsity prior on the dataset-average bucket usage.

    The principled repair of the per-input collapse. :class:`PerInputVariationalEMKSelector`
    charges a full Dirichlet KL against ONE point of evidence per input, so the prior dominates
    and the weights freeze (Basis B reduces to Basis A). Here the per-input weights
    ``w(x) = softmax(weight_net(x))`` are point estimates driven by the conditional-mixture
    likelihood ``Σ_c w_c(x) φ_c(y|x)`` — which genuinely varies with ``x`` — and parsimony comes
    from a single ``Dir(α₀)`` log-prior on the mean usage ``ū = mean_x w(x)``. That one prior is
    weighed against all ``N`` points (it is divided by ``N`` when the likelihood is averaged),
    matching the 1-prior-vs-``N``-points balance that made Basis A's global Dirichlet work.

    The bins default to FIXED tiles (``adaptive_bin_means=False``) so the only way to move mass
    as ``x`` changes is through the weights — the faithful "classifier over fixed bins" design.
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
        adaptive_bin_means: bool = False,
    ) -> None:
        """Initializes the weight net, the (fixed or adaptive) bin components, and the bypass."""
        super().__init__()
        self.k_max = k_max
        self.K = k_max + 1
        self.alpha0 = float(alpha0)
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        self.adaptive_bin_means = adaptive_bin_means

        self.weight_net = vem._MLP(input_dim, self.K, hidden)
        self.bypass_head = vem._MLP(input_dim, 2, hidden)
        if adaptive_bin_means:
            self.bin_heads = nn.ModuleList([vem._MLP(input_dim, 2, hidden) for _ in range(k_max)])
            if bin_centroids is not None:
                for c, head in enumerate(self.bin_heads):
                    with torch.no_grad():
                        head.net[-1].bias[0] = float(bin_centroids[c])
                        head.net[-1].bias[1] = 0.0
        else:
            centers = np.zeros(k_max, dtype=np.float32) if bin_centroids is None else np.asarray(bin_centroids, dtype=np.float32)
            self.register_buffer("bin_means", torch.as_tensor(centers))
            self.bin_log_vars = nn.Parameter(torch.full((k_max,), float(marginal_log_var)))
        with torch.no_grad():
            self.bypass_head.net[-1].bias[0] = float(marginal_mean)
            self.bypass_head.net[-1].bias[1] = float(marginal_log_var)

    def _clamp_lv(self, lv: torch.Tensor) -> torch.Tensor:
        """Clamps a log-variance for numerical stability."""
        return lv.clamp(self.log_var_min, self.log_var_max)

    def per_class_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns per-class ``(mu, log_var)``, each ``(N, K)`` with the bypass last."""
        n = x.shape[0]
        if self.adaptive_bin_means:
            mus, lvs = [], []
            for c in range(self.k_max):
                out = self.bin_heads[c](x)
                mus.append(out[:, 0])
                lvs.append(self._clamp_lv(out[:, 1]))
            bin_mu = torch.stack(mus, dim=1)
            bin_lv = torch.stack(lvs, dim=1)
        else:
            bin_mu = self.bin_means.unsqueeze(0).expand(n, -1)
            bin_lv = self._clamp_lv(self.bin_log_vars).unsqueeze(0).expand(n, -1)
        byp = self.bypass_head(x)
        mu = torch.cat([bin_mu, byp[:, 0:1]], dim=1)
        log_var = torch.cat([bin_lv, self._clamp_lv(byp[:, 1:2])], dim=1)
        return mu, log_var

    def log_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-input log mixing weights ``log softmax(weight_net(x))``, shape ``(N, K)``."""
        return torch.log_softmax(self.weight_net(x), dim=-1)

    @torch.no_grad()
    def weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-input mixing weights ``softmax(weight_net(x))``, shape ``(N, K)``."""
        return torch.softmax(self.weight_net(x), dim=-1)

    def recon_per_point(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Per-point conditional-mixture log-likelihood ``log Σ_c w_c(x) φ_c(y|x)``, shape ``(N,)``."""
        mu, log_var = self.per_class_params(x)
        log_phi = vem.gaussian_log_density(y, mu, log_var)
        return torch.logsumexp(self.log_weights(x) + log_phi, dim=-1)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative MAP objective: mean NLL minus the single aggregate-usage log-prior over ``N``."""
        recon = self.recon_per_point(x, y).mean()
        u_bar = torch.softmax(self.weight_net(x), dim=-1).mean(dim=0)
        log_prior = (self.alpha0 - 1.0) * torch.log(u_bar.clamp_min(1e-12)).sum()
        return -recon - log_prior / x.shape[0]

    @torch.no_grad()
    def mixture_nll_per_point(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Per-point NLL under the per-input mixture, shape ``(N,)``."""
        return -self.recon_per_point(x, y)


def train_aggregate_sparsity(
    x: np.ndarray,
    y: np.ndarray,
    k_max: int,
    alpha0: float = 0.1,
    n_epochs: int = 800,
    lr: float = 1e-2,
    hidden: int = 32,
    adaptive_bin_means: bool = False,
    device: str = "cpu",
    seed: int = 0,
) -> AggregateSparsityKSelector:
    """Fits an :class:`AggregateSparsityKSelector` by full-batch gradient on the MAP objective.

    Args:
        x: features, shape ``(N, D)`` or ``(N,)``.
        y: targets, shape ``(N,)``.
        k_max: number of discretisation classes (bins).
        alpha0: Dirichlet prior concentration on the mean usage (``< 1`` to favour sparsity).
        n_epochs: gradient iterations.
        lr: Adam learning rate.
        hidden: hidden width.
        adaptive_bin_means: see :class:`AggregateSparsityKSelector` (default fixed tiles).
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

    model = AggregateSparsityKSelector(
        input_dim=x_t.shape[1],
        k_max=k_max,
        alpha0=alpha0,
        bin_centroids=centroids,
        marginal_mean=float(y_arr.mean()),
        marginal_log_var=float(np.log(max(y_arr.var(), 1e-4))),
        hidden=hidden,
        adaptive_bin_means=adaptive_bin_means,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = model.loss(x_t, y_t)
        loss.backward()
        opt.step()
    return model
