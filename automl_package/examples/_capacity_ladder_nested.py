"""Nested-k surrogate (capacity-ladder task K4): the VALID k-component mixture ladder.

R1 (`capacity_ladder_results/R1_verdict.md`) diagnosed the K1/K2/K3 catastrophe as an
artifact of *prefix-masking an un-ordered mixture*: the June `AggregateSparsityKSelector`
was trained as a full mixture with no nesting pressure, so "the first c components" was an
arbitrary subset (dead bins, scrambled locations) and the prefix ladder was invalid. K4 is
the construction that makes the prefix ladder valid: one surrogate trained with per-sample
``k ~ Uniform{1..k_max}`` whose loss is the NLL of the renormalized mixture over ONLY the
first ``k`` components. That puts every prefix directly in the training objective, forcing
"the first c components" to be a genuine c-component mixture for every c — an importance
ordering created by nesting, not seeded by location.

Design rules (autocast rank-ladder §3/§6, imported by the execution plan):
  * Uniform draws are a SCHEDULE, not a prior — no evidence-balance audit, no tuned lambda,
    no K_PENALTY (strictly-probabilistic premise). Component 1 (index 0) is the k=1 rung =
    the direct/bypass single Gaussian (seeded at the marginal); nesting differentiates the
    rest.
  * No k input to the network — nesting replaces conditioning; the weight net sees only x.
  * Evaluation at ALL k from ONE forward pass, cached into the (N, k_max) score table via
    :meth:`NestedKSurrogate.nested_components`, whose prefix renormalization is byte-for-byte
    the one `_capacity_ladder.score_table` applies — the trained ladder IS the read ladder.

This module is the K4 instrument; K5 reads the per-input curve off the same table and cross-
checks it against the June arbiter (R1 amendment C).
"""

from __future__ import annotations

import math

import _capacity_ladder as cl  # score_table (read-vs-train consistency selftest)
import _variational_em as vem  # _MLP, gaussian_log_density
import numpy as np
import torch
import torch.nn as nn


class NestedKSurrogate(nn.Module):
    """One over-provisioned surrogate trained as a nested prefix of k-component mixtures.

    ``k_max`` per-class Gaussian heads plus a weight net over the same ``k_max`` components
    (NO separate bypass class: rung 1 = component 0 is the direct single Gaussian). The
    prefix of the first ``c`` components, renormalized by masked softmax over their weight
    logits, is scored as a c-component Gaussian mixture — identically at train time
    (:meth:`masked_prefix_nll`, per-sample drawn ``c``) and at read time
    (:meth:`nested_components` → `_capacity_ladder.score_table`).
    """

    def __init__(
        self,
        input_dim: int,
        k_max: int,
        marginal_mean: float = 0.0,
        marginal_log_var: float = 0.0,
        hidden: int = 32,
        log_var_min: float = -8.0,
        log_var_max: float = 4.0,
        adaptive_bin_means: bool = True,
        bin_centroids: np.ndarray | None = None,
    ) -> None:
        """Initializes the weight net and the per-component heads.

        Args:
            input_dim: feature dimension.
            k_max: number of mixture components (the ladder's top rung; rung grid 1..k_max).
            marginal_mean: bias init for the k=1 rung's mean (the direct/bypass Gaussian) when
                `bin_centroids` is None.
            marginal_log_var: bias init for every head's log-variance (a fair marginal start).
            hidden: hidden width of the sub-networks.
            log_var_min: lower clamp on predicted log-variances (variance-collapse guard).
            log_var_max: upper clamp on predicted log-variances.
            adaptive_bin_means: if True (default, the K4 design) each component is an
                x-dependent head; if False the component means are fixed at 0 with per-
                component learnable widths (kept only for ablation parity with the June infra).
            bin_centroids: optional length-`k_max` spread seed for the adaptive heads' means
                (e.g. y-percentiles). Seeding every component at a distinct location breaks the
                component-STARVATION symmetry — with only component 0 seeded (`None`) the weight
                net collapses onto that one head on multimodal data for some seeds (measured on
                toys D/E). This is an INIT ONLY (no penalty, no tuned λ); the nested training
                still imposes the importance ordering on top of the spread start, so it does not
                reintroduce the R1 location-order pathology (which was a static prefix-READ, not
                a trained nesting).
        """
        super().__init__()
        self.k_max = int(k_max)
        self.log_var_min = float(log_var_min)
        self.log_var_max = float(log_var_max)
        self.adaptive_bin_means = bool(adaptive_bin_means)

        self.weight_net = vem._MLP(input_dim, self.k_max, hidden)
        if self.adaptive_bin_means:
            self.component_heads = nn.ModuleList([vem._MLP(input_dim, 2, hidden) for _ in range(self.k_max)])
            with torch.no_grad():
                if bin_centroids is not None:
                    # Spread seed: every head owns a distinct region from the start (anti-starvation).
                    for c, head in enumerate(self.component_heads):
                        head.net[-1].bias[0] = float(bin_centroids[c])
                        head.net[-1].bias[1] = float(marginal_log_var)
                else:
                    # Marginal-only seed: k=1 rung at the marginal, the rest at random init.
                    self.component_heads[0].net[-1].bias[0] = float(marginal_mean)
                    self.component_heads[0].net[-1].bias[1] = float(marginal_log_var)
        else:
            self.register_buffer("component_means", torch.zeros(self.k_max))
            self.component_log_vars = nn.Parameter(torch.full((self.k_max,), float(marginal_log_var)))

    def _clamp_lv(self, lv: torch.Tensor) -> torch.Tensor:
        """Clamps a log-variance into the numerical-stability band."""
        return lv.clamp(self.log_var_min, self.log_var_max)

    def component_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-component ``(mu, log_var)``, each ``(N, k_max)`` (component 0 = the k=1 rung)."""
        if self.adaptive_bin_means:
            mus, lvs = [], []
            for c in range(self.k_max):
                out = self.component_heads[c](x)
                mus.append(out[:, 0])
                lvs.append(self._clamp_lv(out[:, 1]))
            return torch.stack(mus, dim=1), torch.stack(lvs, dim=1)
        n = x.shape[0]
        mu = self.component_means.unsqueeze(0).expand(n, -1)
        lv = self._clamp_lv(self.component_log_vars).unsqueeze(0).expand(n, -1)
        return mu, lv

    def weight_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalized per-component weight logits ``(N, k_max)`` (the prefix is masked later)."""
        return self.weight_net(x)

    def masked_prefix_nll(self, x: torch.Tensor, y: torch.Tensor, k_draw: torch.Tensor) -> torch.Tensor:
        """Mean per-sample NLL of the renormalized mixture over each sample's first ``k_draw`` components.

        For sample i the active prefix is components ``0..k_draw[i]-1``; the weights are
        ``softmax`` over the active logits (inactive logits masked to ``-inf`` → weight 0), and
        the score is the log of that renormalized ``k_draw[i]``-component Gaussian mixture. This
        is exactly the prefix renormalization `_capacity_ladder.score_table` applies at read time.

        Args:
            x: features, ``(N, D)``.
            y: targets, ``(N,)`` or ``(N, 1)``.
            k_draw: per-sample prefix length, ``(N,)`` ints in ``1..k_max``.

        Returns:
            Scalar mean NLL over the batch.
        """
        logits = self.weight_logits(x)  # (N, k_max)
        mu, log_var = self.component_params(x)
        log_phi = vem.gaussian_log_density(y, mu, log_var)  # (N, k_max)

        arange = torch.arange(self.k_max, device=x.device).unsqueeze(0)  # (1, k_max)
        active = arange < k_draw.unsqueeze(1)  # (N, k_max) True on the drawn prefix
        masked_logits = logits.masked_fill(~active, float("-inf"))
        log_w = masked_logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)  # log_softmax over prefix
        recon = torch.logsumexp(log_w + log_phi, dim=1)  # inactive terms are -inf → excluded
        return -recon.mean()

    @torch.no_grad()
    def nested_components(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The `_capacity_ladder.NestedComponentsModel` read interface: cached per-component params.

        Args:
            x: features, ``(N, D)`` or ``(N,)`` numpy.

        Returns:
            ``(logits, means, log_vars)`` numpy arrays, each ``(N, k_max)``. `score_table`
            renormalizes the first-c prefix of ``logits`` and scores the c-component mixture —
            the same masked softmax used in :meth:`masked_prefix_nll`.
        """
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        device = next(self.parameters()).device
        x_t = torch.as_tensor(x_arr, device=device)
        logits = self.weight_logits(x_t)
        mu, log_var = self.component_params(x_t)
        return logits.cpu().numpy(), mu.cpu().numpy(), log_var.cpu().numpy()


def train_nested_k_surrogate(
    x: np.ndarray,
    y: np.ndarray,
    k_max: int,
    n_epochs: int = 800,
    lr: float = 1e-2,
    hidden: int = 32,
    adaptive_bin_means: bool = True,
    device: str = "cpu",
    seed: int = 0,
    sandwich: bool = False,
    spread_init: bool = True,
    fixed_k: int | None = None,
) -> NestedKSurrogate:
    """Fits a :class:`NestedKSurrogate` by full-batch gradient on the nested-prefix NLL.

    Every epoch draws a FRESH per-sample ``k ~ Uniform{1..k_max}`` (a schedule, re-drawn each
    step) and minimizes the mean renormalized-prefix NLL over those draws. No M-step, no
    penalty, no prior.

    Args:
        x: features, ``(N, D)`` or ``(N,)``.
        y: targets, ``(N,)``.
        k_max: number of components / top rung.
        n_epochs: full-batch gradient steps.
        lr: Adam learning rate.
        hidden: hidden width.
        adaptive_bin_means: adaptive x-dependent heads (K4 design) if True.
        device: torch device string.
        seed: RNG seed for init AND the per-epoch draws (reproducible).
        sandwich: registered ordering-fallback arm (i) — if True, force k=1 and k=k_max present
            every epoch by overwriting the first/last draws (cheap; a schedule, not a loss term).
        spread_init: seed the component means at ``k_max`` even y-quantiles of ``y`` (default,
            the anti-starvation fix — see :class:`NestedKSurrogate`). ``False`` seeds only the
            k=1 rung at the marginal, which collapses for some seeds on multimodal data.
        fixed_k: if set, draw ``k ≡ fixed_k`` every epoch instead of ``Uniform{1..k_max}`` — i.e.
            train a pure fixed-``k`` mixture in the SAME architecture, NO nesting. This is the
            same-architecture B-coh control: nested-at-rung-k vs this isolates the nesting cost
            from the surrogate-vs-MDN architecture gap.

    Returns:
        The fitted model.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    torch.manual_seed(int(seed))

    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=np.float32).ravel()
    x_t = torch.as_tensor(x_arr, device=device)
    y_t = torch.as_tensor(y_arr, device=device)
    n = x_t.shape[0]

    centroids = None
    if spread_init and adaptive_bin_means:
        quantiles = np.linspace(0.0, 100.0, k_max + 2)[1:-1]  # k_max interior quantiles (June convention)
        centroids = np.percentile(y_arr, quantiles)

    model = NestedKSurrogate(
        input_dim=x_t.shape[1],
        k_max=k_max,
        marginal_mean=float(y_arr.mean()),
        marginal_log_var=float(np.log(max(y_arr.var(), 1e-4))),
        hidden=hidden,
        adaptive_bin_means=adaptive_bin_means,
        bin_centroids=centroids,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(int(n_epochs)):
        model.train()
        opt.zero_grad()
        k_draw = torch.full((n,), int(fixed_k), dtype=torch.long, device=device) if fixed_k is not None else torch.randint(1, k_max + 1, (n,), generator=gen).to(device)
        if sandwich and fixed_k is None and n >= 2:
            k_draw[0] = 1
            k_draw[-1] = k_max
        loss = model.masked_prefix_nll(x_t, y_t, k_draw)
        loss.backward()
        opt.step()
    return model


def _selftest() -> bool:
    """Plumbing de-risk: train-time prefix NLL must match the read-time score-table column.

    Checks, on a tiny fit:
      1. `nested_components` shapes are ``(N, k_max)``.
      2. The full-prefix (k=k_max) training NLL equals ``-mean`` of `score_table`'s last column
         (train and read renormalize the SAME way) — the R1 correctness the whole lane rests on.
      3. The k=1 column of the score table is the single Gaussian log-likelihood of component 0.
      4. Training reduces the (fixed-draw) nested loss.
    """
    rng = np.random.default_rng(0)
    n, k_max = 200, 4
    x = rng.uniform(-2.0, 2.0, size=(n, 1)).astype(np.float32)
    y = (np.sin(x[:, 0]) + 0.2 * rng.standard_normal(n)).astype(np.float32)

    model = train_nested_k_surrogate(x, y, k_max=k_max, n_epochs=150, lr=1e-2, seed=0)

    logits, means, log_vars = model.nested_components(x)
    assert logits.shape == (n, k_max), "nested_components logits shape"
    assert means.shape == (n, k_max), "nested_components means shape"
    assert log_vars.shape == (n, k_max), "nested_components log_vars shape"

    # (2) read-vs-train consistency at the full prefix.
    table = cl.score_table(model, x, y, c_grid=list(range(1, k_max + 1)))  # (N, k_max)
    read_full = float(table[:, -1].mean())
    with torch.no_grad():
        x_t = torch.as_tensor(x)
        y_t = torch.as_tensor(y)
        k_full = torch.full((n,), k_max, dtype=torch.long)
        train_full = -float(model.masked_prefix_nll(x_t, y_t, k_full))
    assert abs(read_full - train_full) < 1e-4, f"train/read prefix mismatch: read={read_full:.6f} train={train_full:.6f}"

    # (3) k=1 column == single Gaussian LL of component 0.
    mu0 = means[:, 0]
    lv0 = log_vars[:, 0]
    single = -0.5 * (math.log(2.0 * math.pi) + lv0 + (y - mu0) ** 2 * np.exp(-lv0))
    assert np.allclose(table[:, 0], single, atol=1e-5), "k=1 column != component-0 single Gaussian"

    # (4) loss decreases under a fixed full-prefix draw.
    m2 = NestedKSurrogate(input_dim=1, k_max=k_max, marginal_mean=float(y.mean()), marginal_log_var=0.0)
    opt = torch.optim.Adam(m2.parameters(), lr=1e-2)
    x_t = torch.as_tensor(x)
    y_t = torch.as_tensor(y)
    k_full = torch.full((n,), k_max, dtype=torch.long)
    first = last = None
    for step in range(80):
        opt.zero_grad()
        loss = m2.masked_prefix_nll(x_t, y_t, k_full)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first, f"nested loss did not decrease: first={first:.4f} last={last:.4f}"

    print(f"[selftest] OK  read_full={read_full:.4f} train_full={train_full:.4f}  loss {first:.4f}->{last:.4f}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
