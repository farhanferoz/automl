"""Frozen residual cascade — width grown one tanh block at a time, each block frozen forever.

(`docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md` §2.1-2.3, §4.1)

K blocks, each a single-hidden-layer tanh subnet of width 1: `h_b(x) = tanh(u_b x + c_b)` with two
scalar readouts `(dmu_b, ds_b) = (a_b h_b + d_b, a'_b h_b + d'_b)`. The rung-k (prefix-k) model is
the ADDITIVE composition `mu_k = sum_{b<k} dmu_b`, `s_k = sum_{b<k} ds_b` — additive in the mean,
additive in the LOG-variance (multiplicative in variance itself), NGBoost's parametrization (plan
§2.1/§2.6). Rung 0 (no blocks) is `(mu, s) = (0, 0) = N(0, 1)` -- exactly the standardized target
marginal, the width lane's analog of the low-rank ladder's diagonal-only rung.

By Lemma 1 (plan §2.2) a sum of k width-1 tanh blocks is EXACTLY a width-k single-hidden-layer tanh
network -- rung k's function class equals `nested_width_net.NestedWidthNet`'s width-k class, with
one extra freedom (an independent readout bias per prefix instead of one shared bias). So the
cascade is not a new architecture family: it is the nested net with (a) a STAGED, FROZEN training
scheme and (b) per-prefix readout biases. `train_cascade` (module-level, below) is that scheme:
stage b zero-inits block b's readouts (so the stage starts EXACTLY at rung b-1's output), trains
ONLY block b's parameters against the cached, frozen rung-(b-1) prefix to convergence
(`convergence.fit_to_convergence`, multiple restarts, keep best-val), applies the acceptance rule
(reset to inert if the stage did not beat rung b-1 by `convergence.DEFAULT_MIN_DELTA`), then FREEZES
block b permanently (plan §2.3 Lemma 2). This guarantees `NLL_val(rung k)` non-increasing in k, every
rung a valid calibrated model, and rung identity stable forever -- the coherence invariant in its
guaranteed (weak) form.

Selftest (`--selftest`, no training, random init only) checks five architectural invariants:
  (a) zero-init -> `forward_width(x, k)` is exactly `(0, 0)` for every k (rung-0 = N(0,1) propagates
      through all-zero readouts, matching the constructor's own zero-init).
  (b) after randomizing readouts: `all_widths_forward` column k-1 == `forward_width(x, k)` (the two
      code paths -- explicit per-block sum, and the vectorized cumsum -- must agree), AND rung-k
      output is invariant to perturbing blocks >= k (the prefix property).
  (c) freeze check -- after `freeze_blocks_below(b)`, a backward pass through the rung-w_max NLL
      leaves `grad is None or ~0` for every frozen-block parameter.
  (d) stage-start identity -- with block b's own readouts zeroed (other blocks arbitrary), rung-b
      output equals rung-(b-1) output exactly (the property `train_cascade` relies on to start each
      stage at the previous rung).
  (e) acceptance-reset -- after writing garbage into block b's readouts then applying the reset
      (zeroing them again, the acceptance rule's rejection path), rung-b == rung-(b-1) again.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/cascade_width_net.py --selftest
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import convergence as cvg  # noqa: E402
import nested_width_net as nwn  # noqa: E402

W_MAX_DEFAULT = 12
LR_DEFAULT = 1e-2  # plan §4.0: LR 1e-2 Adam everywhere.
RESTARTS_DEFAULT = 3  # plan §2.3: multi-restart per stage (a single tanh unit is seed-sensitive).
BETA_NLL_BETA = 0.5  # Seitzer et al. 2022 recommended beta; only active when `beta_nll=True`.

_ALL_WIDTHS_TOL = 1e-5
_PREFIX_PERTURB_TOL = 1e-6
_STAGE_IDENTITY_TOL = 1e-6
_FREEZE_GRAD_TOL = 1e-8
_GARBAGE_SANITY_TOL = 1e-3  # selftest (e): garbage perturbation must actually move the output by more than this


def _zero_readouts_(block: nn.ModuleDict) -> None:
    """In-place zeros a block's `mean_head`/`logvar_head` weight AND bias (the stage-start / inert state)."""
    with torch.no_grad():
        block["mean_head"].weight.zero_()
        block["mean_head"].bias.zero_()
        block["logvar_head"].weight.zero_()
        block["logvar_head"].bias.zero_()


class ResidualCascadeNet(nn.Module):
    """`w_max` frozen residual blocks; rung k is the additive prefix sum of blocks `0..k-1`.

    Block b (0-indexed) is `nn.ModuleDict({"trunk": Linear(1,1), "act": activation(), "mean_head":
    Linear(1,1), "logvar_head": Linear(1,1)})`. Every block's readouts start at exactly zero (trunk
    keeps its default init) so an untrained net's every rung is the N(0,1) marginal -- selftest (a).
    """

    def __init__(self, w_max: int = W_MAX_DEFAULT, activation: type[nn.Module] = nn.Tanh) -> None:
        """Builds `w_max` width-1 tanh blocks, readouts zero-initialized.

        Args:
            w_max: number of blocks (== the largest rung this net can express).
            activation: hidden-unit nonlinearity class (instantiated with no args); tanh per the
                plan's fixed hyperparameter (§4.0).
        """
        super().__init__()
        self.w_max = int(w_max)
        self.blocks = nn.ModuleList(
            nn.ModuleDict({"trunk": nn.Linear(1, 1), "act": activation(), "mean_head": nn.Linear(1, 1), "logvar_head": nn.Linear(1, 1)})
            for _ in range(self.w_max)
        )
        for block in self.blocks:
            _zero_readouts_(block)

    def block_contrib(self, x: torch.Tensor, b: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Block `b`'s (0-indexed) OWN `(dmu, ds)` readout pair, each shape `(N, 1)`."""
        block = self.blocks[b]
        h = block["act"](block["trunk"](x))
        return block["mean_head"](h), block["logvar_head"](h)

    def forward_width(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at rung `k` (1..w_max): the additive prefix sum of blocks `0..k-1`."""
        if not (1 <= k <= self.w_max):
            raise ValueError(f"k={k} out of range [1, {self.w_max}]")
        mu = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        s = torch.zeros_like(mu)
        for b in range(k):
            dmu, ds = self.block_contrib(x, b)
            mu = mu + dmu
            s = s + ds
        return mu, s

    def all_widths_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at every rung `k=1..w_max` in one pass; each shape `(N, w_max)`.

        Column `k-1` is rung-k's readout: each block's own contribution stacked into a `(N, w_max)`
        table, then `torch.cumsum` over blocks -- the additive-prefix property made vectorized (same
        trick as `nested_width_net.NestedWidthNet.all_widths_forward`, here over BLOCKS instead of
        masked hidden nodes, since each block already computes its own delta independently).
        """
        dmu_cols, ds_cols = [], []
        for b in range(self.w_max):
            dmu, ds = self.block_contrib(x, b)
            dmu_cols.append(dmu)
            ds_cols.append(ds)
        dmu_stack = torch.cat(dmu_cols, dim=1)  # (N, w_max)
        ds_stack = torch.cat(ds_cols, dim=1)
        return torch.cumsum(dmu_stack, dim=1), torch.cumsum(ds_stack, dim=1)

    def freeze_blocks_below(self, b: int) -> None:
        """`requires_grad_(False)` on blocks with (0-indexed) index `< b` -- the frozen prefix."""
        for idx in range(b):
            self.blocks[idx].requires_grad_(False)


def _stage_nll(
    net: ResidualCascadeNet, block_idx: int, mu_prev: torch.Tensor, s_prev: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One stage's per-example NLL and its `(mu_total, s_total)`, using the CACHED frozen prefix (no re-run)."""
    dmu, ds = net.block_contrib(x, block_idx)
    mu_total = mu_prev + dmu
    s_total = s_prev + ds
    nll = -nwn.gaussian_log_likelihood(mu_total.squeeze(1), s_total.squeeze(1), y)
    return nll, s_total


def train_cascade(
    net: ResidualCascadeNet,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    seed: int,
    restarts: int = RESTARTS_DEFAULT,
    max_epochs: int,
    check_every: int,
    patience: int,
    min_delta: float,
    lr: float = LR_DEFAULT,
    beta_nll: bool = False,
) -> dict[int, dict]:
    """Trains the cascade stagewise (plan §2.3): stage b trains ONLY block b against the frozen prefix.

    For each stage b = 1..w_max (block index b-1): the rung-(b-1) prefix is cached once (under
    `no_grad`, zeros for b=1) so restarts never re-run the already-frozen blocks. Each of `restarts`
    restarts re-inits block b's trunk (fresh init, seeded `seed*1000 + b*10 + restart`) and zeros its
    readouts, then trains block-b's own parameters ONLY (`cvg.fit_to_convergence`) against the
    cached-prefix NLL. The restart with the lowest `best_val` wins. Acceptance rule: if the winning
    `best_val` did not beat rung-(b-1)'s val NLL by more than `cvg.DEFAULT_MIN_DELTA`, the block's
    readouts are reset to zero (rung b becomes inert, `accepted=False`); otherwise its trained
    weights are kept (`accepted=True`). Either way block b is frozen (`freeze_blocks_below`) before
    the next stage.

    `beta_nll`: if True, the STAGE TRAINING loss (not the held-out val loss, which is always plain
    NLL per §2.5's "the bar never moves") is per-point NLL times `stop_grad(exp(s_total)**0.5)`
    (Seitzer et al. 2022, beta=0.5) -- wired now so the §2.5 escalation trigger is a rerun, not a
    code change; default off (plain NLL, comparable to every other W battery).

    Args:
        net: the `ResidualCascadeNet` to train in place.
        x_tr: standardized training inputs, `(N, 1)` or broadcastable.
        y_tr: standardized training targets, `(N,)` or broadcastable.
        x_val: standardized held-out inputs, used only for convergence monitoring and the
            acceptance rule.
        y_val: standardized held-out targets, used only for convergence monitoring and the
            acceptance rule.
        seed: RNG seed for reproducible per-stage/per-restart re-init.
        restarts: random restarts per stage (fresh trunk init each; §2.3).
        max_epochs: safety cap passed straight through to `cvg.fit_to_convergence` for each
            restart's own convergence gate.
        check_every: epochs between held-out checkpoints, passed through to `cvg.fit_to_convergence`.
        patience: flat checkpoints that declare a restart converged, passed through to
            `cvg.fit_to_convergence`.
        min_delta: held-out-loss decrease (nats) counted as improvement, passed through to
            `cvg.fit_to_convergence` (NOT the acceptance-rule threshold, which always uses
            `cvg.DEFAULT_MIN_DELTA` per §2.3).
        lr: Adam learning rate (plan §4.0 default 1e-2).
        beta_nll: use the beta-NLL (beta=0.5) STAGE TRAINING loss instead of plain NLL.

    Returns:
        `{stage b -> {"conv": cvg.ConvergenceResult (winning restart), "accepted": bool,
        "val_nll_prev": float}}` for b = 1..w_max.
    """
    x_tr = x_tr.reshape(-1, 1)
    x_val = x_val.reshape(-1, 1)
    y_tr = y_tr.reshape(-1)
    y_val = y_val.reshape(-1)
    n_tr, n_val = x_tr.shape[0], x_val.shape[0]

    results: dict[int, dict] = {}
    for b in range(1, net.w_max + 1):
        block_idx = b - 1
        with torch.no_grad():
            if b == 1:
                mu_prev_tr = torch.zeros(n_tr, 1)
                s_prev_tr = torch.zeros(n_tr, 1)
                mu_prev_val = torch.zeros(n_val, 1)
                s_prev_val = torch.zeros(n_val, 1)
            else:
                mu_prev_tr, s_prev_tr = net.forward_width(x_tr, b - 1)
                mu_prev_val, s_prev_val = net.forward_width(x_val, b - 1)
            val_nll_prev = float((-nwn.gaussian_log_likelihood(mu_prev_val.squeeze(1), s_prev_val.squeeze(1), y_val)).mean().item())

        block = net.blocks[block_idx]
        best_result: cvg.ConvergenceResult | None = None
        best_state: dict | None = None
        for restart in range(restarts):
            torch.manual_seed(seed * 1000 + b * 10 + restart)
            block["trunk"] = nn.Linear(1, 1)  # fresh hidden-weight init this restart
            _zero_readouts_(block)
            opt = torch.optim.Adam(block.parameters(), lr=lr)

            def step_fn(
                block_idx: int = block_idx, opt: torch.optim.Optimizer = opt, mu_prev_tr: torch.Tensor = mu_prev_tr, s_prev_tr: torch.Tensor = s_prev_tr
            ) -> None:
                opt.zero_grad()
                nll, s_total = _stage_nll(net, block_idx, mu_prev_tr, s_prev_tr, x_tr, y_tr)
                if beta_nll:
                    weight = torch.exp(s_total.squeeze(1)).pow(BETA_NLL_BETA).detach()
                    loss = (weight * nll).mean()
                else:
                    loss = nll.mean()
                loss.backward()
                opt.step()

            def val_loss_fn(block_idx: int = block_idx, mu_prev_val: torch.Tensor = mu_prev_val, s_prev_val: torch.Tensor = s_prev_val) -> float:
                with torch.no_grad():
                    nll, _s_total = _stage_nll(net, block_idx, mu_prev_val, s_prev_val, x_val, y_val)
                    return float(nll.mean().item())

            conv_result = cvg.fit_to_convergence(block, step_fn, val_loss_fn, max_epochs=max_epochs, check_every=check_every, patience=patience, min_delta=min_delta)
            if best_result is None or conv_result.best_val < best_result.best_val:
                best_result = conv_result
                best_state = {n: t.detach().clone() for n, t in block.state_dict().items()}

        block.load_state_dict(best_state)
        accepted = best_result.best_val < val_nll_prev - cvg.DEFAULT_MIN_DELTA
        if not accepted:
            _zero_readouts_(block)  # rung b becomes inert: identical to rung b-1 (trunk value now irrelevant)

        net.freeze_blocks_below(b)
        results[b] = {"conv": best_result, "accepted": accepted, "val_nll_prev": val_nll_prev}

    net.eval()
    return results


# ---------------------------------------------------------------------------
# Selftest -- random-init only, no training. MUST pass before any real read.
# ---------------------------------------------------------------------------


def _assert_zero_init_is_marginal(net: ResidualCascadeNet, x: torch.Tensor) -> tuple[bool, float]:
    """(a) zero-init -> forward_width(x, k) is exactly (0, 0) for every k."""
    ok_all = True
    max_err = 0.0
    with torch.no_grad():
        for k in range(1, net.w_max + 1):
            mu, s = net.forward_width(x, k)
            err = max(mu.abs().max().item(), s.abs().max().item())
            max_err = max(max_err, err)
            ok_all = ok_all and (err == 0.0)
    return ok_all, max_err


def _randomize_readouts_(net: ResidualCascadeNet, generator: torch.Generator) -> None:
    with torch.no_grad():
        for block in net.blocks:
            block["mean_head"].weight.copy_(torch.randn(block["mean_head"].weight.shape, generator=generator))
            block["mean_head"].bias.copy_(torch.randn(block["mean_head"].bias.shape, generator=generator))
            block["logvar_head"].weight.copy_(torch.randn(block["logvar_head"].weight.shape, generator=generator) * 0.1)
            block["logvar_head"].bias.copy_(torch.randn(block["logvar_head"].bias.shape, generator=generator) * 0.1)


def _assert_all_widths_and_prefix(net: ResidualCascadeNet, x: torch.Tensor) -> tuple[bool, float, float]:
    """(b) all_widths_forward == per-k forward_width (tol 1e-5); prefix invariance to blocks >= k (tol 1e-6)."""
    with torch.no_grad():
        mu_all, s_all = net.all_widths_forward(x)
    max_err_consistency = 0.0
    for k in range(1, net.w_max + 1):
        with torch.no_grad():
            mu_k, s_k = net.forward_width(x, k)
        err = max((mu_all[:, k - 1 : k] - mu_k).abs().max().item(), (s_all[:, k - 1 : k] - s_k).abs().max().item())
        max_err_consistency = max(max_err_consistency, err)
    ok_consistency = max_err_consistency < _ALL_WIDTHS_TOL

    max_err_prefix = 0.0
    for k in range(1, net.w_max + 1):
        with torch.no_grad():
            mu0, s0 = net.forward_width(x, k)
        snapshots = []
        if k < net.w_max:
            for b in range(k, net.w_max):
                block = net.blocks[b]
                snapshots.append((block["mean_head"].weight.detach().clone(), block["mean_head"].bias.detach().clone()))
                with torch.no_grad():
                    block["mean_head"].weight.add_(torch.randn_like(block["mean_head"].weight) * 5.0)
                    block["mean_head"].bias.add_(torch.randn_like(block["mean_head"].bias) * 5.0)
        with torch.no_grad():
            mu1, s1 = net.forward_width(x, k)
        for b, (w_orig, b_orig) in zip(range(k, net.w_max), snapshots, strict=True):
            block = net.blocks[b]
            with torch.no_grad():
                block["mean_head"].weight.copy_(w_orig)
                block["mean_head"].bias.copy_(b_orig)
        err = max((mu0 - mu1).abs().max().item(), (s0 - s1).abs().max().item())
        max_err_prefix = max(max_err_prefix, err)
    ok_prefix = max_err_prefix < _PREFIX_PERTURB_TOL

    return (ok_consistency and ok_prefix), max_err_consistency, max_err_prefix


def _assert_freeze(net: ResidualCascadeNet, x: torch.Tensor, y: torch.Tensor, b: int) -> tuple[bool, float]:
    """(c) after freeze_blocks_below(b), a backward pass through rung-w_max NLL zeros frozen-block grads."""
    net.freeze_blocks_below(b)
    for block in net.blocks:
        for p in block.parameters():
            p.grad = None
    mu, s = net.forward_width(x, net.w_max)
    nll = -nwn.gaussian_log_likelihood(mu.squeeze(1), s.squeeze(1), y)
    nll.mean().backward()
    max_grad = 0.0
    for idx in range(b):
        for p in net.blocks[idx].parameters():
            if p.grad is not None:
                max_grad = max(max_grad, p.grad.abs().max().item())
    return max_grad < _FREEZE_GRAD_TOL, max_grad


def _assert_stage_start_identity(net: ResidualCascadeNet, x: torch.Tensor, b: int) -> tuple[bool, float]:
    """(d) with block b's (1-indexed) own readouts zeroed, rung-b output equals rung-(b-1) exactly."""
    _zero_readouts_(net.blocks[b - 1])
    with torch.no_grad():
        mu_b, s_b = net.forward_width(x, b)
        mu_prev, s_prev = net.forward_width(x, b - 1)
    err = max((mu_b - mu_prev).abs().max().item(), (s_b - s_prev).abs().max().item())
    return err < _STAGE_IDENTITY_TOL, err


def _assert_acceptance_reset(net: ResidualCascadeNet, x: torch.Tensor, generator: torch.Generator, b: int) -> tuple[bool, float]:
    """(e) garbage into block b's readouts, then the reset (rejection path) -> rung-b == rung-(b-1) again."""
    block = net.blocks[b - 1]
    with torch.no_grad():
        block["mean_head"].weight.copy_(torch.randn(block["mean_head"].weight.shape, generator=generator) * 10.0)
        block["mean_head"].bias.copy_(torch.randn(block["mean_head"].bias.shape, generator=generator) * 10.0)
        block["logvar_head"].weight.copy_(torch.randn(block["logvar_head"].weight.shape, generator=generator) * 10.0)
        block["logvar_head"].bias.copy_(torch.randn(block["logvar_head"].bias.shape, generator=generator) * 10.0)
    with torch.no_grad():
        mu_garbage, _s_garbage = net.forward_width(x, b)
        mu_prev, s_prev = net.forward_width(x, b - 1)
    garbage_differs = (mu_garbage - mu_prev).abs().max().item() > _GARBAGE_SANITY_TOL  # sanity: garbage actually changed the output

    _zero_readouts_(block)  # the acceptance rule's reset
    with torch.no_grad():
        mu_b, s_b = net.forward_width(x, b)
    err = max((mu_b - mu_prev).abs().max().item(), (s_b - s_prev).abs().max().item())
    return (garbage_differs and err < _STAGE_IDENTITY_TOL), err


def run_selftest() -> bool:
    """Runs all five no-training checks on a randomly-initialized net and prints PASS/FAIL."""
    torch.manual_seed(0)
    net = ResidualCascadeNet(w_max=6)
    net.eval()
    x = torch.randn(37, 1)
    y = torch.randn(37)
    gen = torch.Generator().manual_seed(1)

    ok_a, err_a = _assert_zero_init_is_marginal(net, x)
    print(f"[cascade selftest] (a) zero-init == N(0,1) marginal: max_abs_err={err_a:.3e}  {'PASS' if ok_a else 'FAIL'}")

    _randomize_readouts_(net, gen)
    ok_b, err_consistency, err_prefix = _assert_all_widths_and_prefix(net, x)
    print(
        f"[cascade selftest] (b) all-widths vs per-k consistency: max_abs_err={err_consistency:.3e} (tol={_ALL_WIDTHS_TOL:.0e}); "
        f"prefix invariance to blocks>=k: max_abs_err={err_prefix:.3e} (tol={_PREFIX_PERTURB_TOL:.0e})  {'PASS' if ok_b else 'FAIL'}"
    )

    ok_c, grad_c = _assert_freeze(net, x, y, b=3)
    print(f"[cascade selftest] (c) freeze_blocks_below(3) zeros frozen grads: max_abs_grad={grad_c:.3e} (tol={_FREEZE_GRAD_TOL:.0e})  {'PASS' if ok_c else 'FAIL'}")

    torch.manual_seed(2)
    net_d = ResidualCascadeNet(w_max=6)
    _randomize_readouts_(net_d, gen)
    ok_d, err_d = _assert_stage_start_identity(net_d, x, b=4)
    print(f"[cascade selftest] (d) stage-start identity (rung4==rung3 with block4 zeroed): max_abs_err={err_d:.3e}  {'PASS' if ok_d else 'FAIL'}")

    torch.manual_seed(3)
    net_e = ResidualCascadeNet(w_max=6)
    _randomize_readouts_(net_e, gen)
    ok_e, err_e = _assert_acceptance_reset(net_e, x, gen, b=4)
    print(f"[cascade selftest] (e) acceptance-reset (garbage then reset -> rung4==rung3): max_abs_err={err_e:.3e}  {'PASS' if ok_e else 'FAIL'}")

    ok = ok_a and ok_b and ok_c and ok_d and ok_e
    print(f"[cascade selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Parses args and runs the selftest, or prints help (this module has no standalone real-run mode)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="No-training known-answer checks of the cascade's architectural invariants.")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
