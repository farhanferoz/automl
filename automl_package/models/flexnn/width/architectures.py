"""The four width-dial architectures certified by the width-MSE programme.

Moved from `automl_package/examples/nested_width_net.py` (2026-07-21,
`docs/plans/capacity_programme/flexnn-package.md` Task FP-2 -- the boundary rule, MASTER Decision
19: `automl_package/models/` holds library architectures, `automl_package/examples/` holds
experiment drivers). `automl_package/examples/nested_width_net.py` is now a re-export shim over
this module (the `automl_package/examples/convergence.py` precedent) and keeps the toy generators
(`make_hetero`, `make_hetero3`), the training schedule (`train_nested_width`), and the selftest CLI
-- those are experiment protocol, not library architecture, and stay on the examples side per this
task's non-goals.

Depth is fixed at a single hidden layer everywhere in this module; the ONLY capacity axis is the
number of ACTIVE hidden nodes ("width"). All four classes share the `(w_max, activation)`
constructor shape and expose `forward_width(x, k)` / `all_widths_forward(x)`, differing only in how
much of the trunk/readout is shared across widths:

- `NestedWidthNet` -- shared trunk, ONE shared mean/logvar head pair (the charter-question arch).
- `IndependentWidthNet` -- `w_max` disjoint one-hidden-layer sub-nets, no weight sharing at all.
- `SharedTrunkPerWidthHeadNet` -- shared trunk, per-width OWN mean head. **The architecture of
  record**: MASTER Decision 1's certified G-WIDTH arch (`docs/width_mse_2026-07-16/
  verdict_variable_width_mse.md` §10 certification addendum).
- `SharedReadoutPerWidthAffineNet` -- shared trunk AND shared readout, plus a 2-parameter per-width
  affine on top (the minimum-seam arm).

No behaviour change from the move: every docstring, formula, and tolerance below is verbatim from
the pre-move file.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from automl_package.utils.capacity_accounting import _linear_macs, executed_flops

W_MAX_DEFAULT = 16


class NestedWidthNet(nn.Module):
    """Single-hidden-layer probabilistic net with a nested-width readout.

    `Linear(1 -> w_max)` -> activation -> hidden vector `h`; `mean_head`/`logvar_head` are
    `Linear(w_max -> 1)`. A width-k forward zeros `h[:, k:]` before both readouts.

    Variance status (`width.md` SS3.7, recorded by WSEL-17 Step 3): `logvar_head` is a
    LEARNED-variance readout from this class's pre-ruling probabilistic lineage. The 2026-07-21
    user ruling forbids fitting it in any new run -- variance is FIXED at the true generator
    value, never learned. New runs train MSE or the fixed-sigma weighted objective only; loss
    selectors that would fit this head (e.g. an NLL loss) are historical-reproduction paths.
    """

    def __init__(self, w_max: int = W_MAX_DEFAULT, activation: type[nn.Module] = nn.Tanh) -> None:
        """Builds the trunk + the two (mean, log-variance) readout heads.

        Args:
            w_max: maximum hidden width (the largest prefix the net can express).
            activation: hidden-layer nonlinearity class (instantiated with no args); tanh per
                the plan's fixed hyperparameter (`docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md`
                §2d config).
        """
        super().__init__()
        self.w_max = int(w_max)
        self.trunk = nn.Linear(1, self.w_max)
        self.activation = activation()
        self.mean_head = nn.Linear(self.w_max, 1)
        self.logvar_head = nn.Linear(self.w_max, 1)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, 1) -> (N, w_max)` post-activation hidden representation (the trunk's full output)."""
        return self.activation(self.trunk(x))

    def forward_width(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at one FIXED width `k` (1..w_max), each shape `(N, 1)`.

        Explicit masking (`h[:, k:] = 0` before both readouts) — the literal statement of the
        width-k prefix property; `all_widths_forward` computes the same values for every k at
        once via an algebraically equivalent cumulative sum (selftest (c) checks the two agree).
        """
        if not (1 <= k <= self.w_max):
            raise ValueError(f"k={k} out of range [1, {self.w_max}]")
        h = self.hidden(x)
        mask = torch.zeros_like(h)
        mask[:, :k] = 1.0
        h_masked = h * mask
        return self.mean_head(h_masked), self.logvar_head(h_masked)

    def all_widths_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at EVERY width `k=1..w_max` in one pass; each shape `(N, w_max)`.

        Column `k-1` is width-k's readout. Both readouts are linear in `h`, so zeroing columns
        `k..` before a `Linear(w_max -> 1)` is exactly `sum_{i<k} h[:, i] * weight[0, i] + bias`
        — a per-node contribution followed by a cumulative sum over the first k nodes. Exact and
        vectorized, no python loop over widths (the width analog of `NestedStrategy.
        all_depth_outputs`'s "one forward pass, every depth" cached-forward trick).
        """
        h = self.hidden(x)  # (N, w_max)
        mean_contrib = h * self.mean_head.weight.squeeze(0)  # (N, w_max), elementwise per-node contribution
        logvar_contrib = h * self.logvar_head.weight.squeeze(0)
        mean_all = torch.cumsum(mean_contrib, dim=1) + self.mean_head.bias
        logvar_all = torch.cumsum(logvar_contrib, dim=1) + self.logvar_head.bias
        return mean_all, logvar_all

    def sampled_widths_forward(self, x: torch.Tensor, widths: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` for exactly `widths` (order preserved, repeats allowed), each `(N, len(widths))`.

        `capacity_programme` WSEL-12: a k-dropout training step samples a SUBSET of widths, not every
        `1..w_max` one — the caller that used to loop `forward_width` once per sampled width (recomputing
        `hidden(x)` every time even though every width shares the same trunk) now calls this instead.
        Uses the SAME explicit-masking formula as `forward_width` (not the `all_widths_forward` cumsum
        shortcut, which is only algebraically -- not bit-for-bit -- equivalent, per selftest (c)'s
        `_CONSISTENCY_TOL=1e-5` tolerance): the training-loop equivalence bar this method exists for is
        bit-for-bit, so it must reuse the identical arithmetic, just with `h` computed once instead of
        once per sampled width.
        """
        h = self.hidden(x)
        means, logvars = [], []
        for k in widths:
            if not (1 <= k <= self.w_max):
                raise ValueError(f"k={k} out of range [1, {self.w_max}]")
            mask = torch.zeros_like(h)
            mask[:, :k] = 1.0
            h_masked = h * mask
            means.append(self.mean_head(h_masked))
            logvars.append(self.logvar_head(h_masked))
        return torch.cat(means, dim=1), torch.cat(logvars, dim=1)

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample width draw: `k` is an `(N,)` int tensor in `[1, w_max]`.

        Gathers each sample's own drawn-width readout out of `all_widths_forward` — the free
        reshape of one forward pass that `NestedStrategy.forward` relies on for depth
        (layer_selection_strategies.py:265-269), here over width columns instead of depth blocks.

        Returns:
            `(mean, log_var)`, each shape `(N, 1)`.
        """
        mean_all, logvar_all = self.all_widths_forward(x)
        col = (k - 1).unsqueeze(1)
        mean = mean_all.gather(1, col)
        logvar = logvar_all.gather(1, col)
        return mean, logvar


class IndependentWidthNet(nn.Module):
    """W4: K DISJOINT one-hidden-layer probabilistic sub-nets, one per width — NO weight sharing.

    The width twin of `automl_package/models/independent_weights_flexible_neural_network.py`
    (`IndependentWeightsFlexibleNN`: independent weights per DEPTH, one `nn.ModuleList` entry per
    depth, trained with the same k-dropout selection). Here width-k is its OWN sub-net
    `Linear(1 -> k) -> tanh -> (mean_head_k, logvar_head_k)`, sharing NOTHING with width-k'≠k. This
    deliberately breaks the shared-trunk prefix property of `NestedWidthNet` — the one variable W1/W2
    pinned as the obstruction (`docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md` §6 follow-up):
    node-0 no longer has to double as a good small-width predictor, so each width is free to fit to
    its own ceiling.

    Exposes the SAME `w_max` / `forward_width` / `all_widths_forward` interface as `NestedWidthNet`,
    so it is a drop-in for `sinc_width_experiment._score_all_widths` and for the SANDWICH branch of
    `train_nested_width` (which touches only `net.w_max` and `net.forward_width`) — the ONLY thing
    that changes vs the shared W2 run is shared-vs-independent weights.

    Variance status (`width.md` SS3.7, recorded by WSEL-17 Step 3): the per-width `logvar_head_k`
    readouts are LEARNED-variance heads from the pre-ruling lineage; fitting them in any new run is
    forbidden (variance is FIXED at the true generator value, never learned). New runs train MSE or
    the fixed-sigma weighted objective only.
    """

    def __init__(self, w_max: int = W_MAX_DEFAULT, activation: type[nn.Module] = nn.Tanh) -> None:
        """Builds `w_max` disjoint sub-nets; sub-net k (0-indexed k-1) has exactly k hidden units."""
        super().__init__()
        self.w_max = int(w_max)
        self.subnets = nn.ModuleList(
            nn.ModuleDict({"trunk": nn.Linear(1, k), "act": activation(), "mean_head": nn.Linear(k, 1), "logvar_head": nn.Linear(k, 1)})
            for k in range(1, self.w_max + 1)
        )

    def forward_width(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` from width-k's OWN sub-net (uses no other width's weights). Each `(N, 1)`."""
        if not (1 <= k <= self.w_max):
            raise ValueError(f"k={k} out of range [1, {self.w_max}]")
        sub = self.subnets[k - 1]
        h = sub["act"](sub["trunk"](x))
        return sub["mean_head"](h), sub["logvar_head"](h)

    def all_widths_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at every width, each `(N, w_max)`; column k-1 is width-k's own sub-net.

        No cumsum trick here (weights are NOT nested/shared) — just each independent sub-net's output
        concatenated. Matches `NestedWidthNet.all_widths_forward`'s output shape/semantics so the
        shared scoring/selector code path is identical.
        """
        means, logvars = [], []
        for k in range(1, self.w_max + 1):
            mean_k, logvar_k = self.forward_width(x, k)
            means.append(mean_k)
            logvars.append(logvar_k)
        return torch.cat(means, dim=1), torch.cat(logvars, dim=1)


class SharedTrunkPerWidthHeadNet(nn.Module):
    """Width-MSE program's Contingency C: `NestedWidthNet` with ONE change -- per-width mean heads.

    Same shared `Linear(1 -> w_max)` trunk and the SAME prefix-masking mechanism as
    `NestedWidthNet.forward_width` (`h[:, k:] = 0` before the readout), but width `k` reads its OWN
    `mean_head_k = Linear(w_max -> 1)` instead of the one `mean_head` shared across every width. This
    isolates the readout-sharing variable ALONE -- mirroring how `IndependentWidthNet` above isolates
    trunk sharing by breaking everything at once -- so a WP-2 comparison against `NestedWidthNet`
    (shared trunk AND shared readout) and `IndependentWidthNet` (neither shared) attributes a pass/fail
    here to the shared READOUT specifically, not to a confound from also changing the readout's
    parameterization (a genuine Matryoshka-style dedicated `Linear(k -> 1)` head per width, as in
    `matryoshka_width_net.MatryoshkaWidthNet`, changes both the sharing AND the parameter count per
    width; kept out of this class on purpose so exactly one variable moves;
    `docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md` WP-2 Contingency C: "the middle rung").

    MSE-only (`docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md` §0: no variance fitting anywhere) --
    `forward_width`/`all_widths_forward` return a `log_var` of zeros to match the `(mean, log_var)`
    interface every caller expects, but it is never read by `_width_mse` and carries no gradient.

    **WSEL-18 fused mode** (`fused_heads=True`, constructor flag, DEFAULT OFF -- `docs/plans/
    capacity_programme/width.md` WSEL-18): replaces the `w_max` disjoint `mean_heads` with ONE
    lower-triangular-masked `(w_max, w_max)` weight tensor + a `(w_max,)` bias vector, turning
    `all_widths_forward`'s `w_max`-iteration python loop into a single matmul. Under the WSEL-14 ALL
    schedule (every width trained every step) this is MATHEMATICALLY EXACT: row `k-1` is masked to
    its own first `k` columns, so it computes exactly what `mean_heads[k-1]` computes off the same
    masked hidden vector, and elementwise Adam updates on one tensor equal per-tensor updates when
    every row gets a real gradient every step. Under a SAMPLING schedule (sandwich/uniform) it is
    NOT exact -- a width not drawn on a given step gets a genuine ZERO gradient in fused mode (an
    executed Adam no-op, advancing its bias-correction bookkeeping) vs NO gradient at all in unfused
    mode (the optimizer never touches that head) -- so callers must refuse fused mode outside ALL
    (enforced in `kdropout_converged_width_experiment.py`, not here: this class has no notion of
    "schedule", only the trainer does).
    """

    def __init__(self, w_max: int = W_MAX_DEFAULT, activation: type[nn.Module] = nn.Tanh, fused_heads: bool = False) -> None:
        """Builds the shared trunk plus either `w_max` independent mean heads or ONE fused readout.

        Args:
            w_max: maximum hidden width (the largest prefix the net can express).
            activation: hidden-layer nonlinearity class (instantiated with no args); tanh per the
                plan's fixed hyperparameter (`docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md` §5).
            fused_heads: WSEL-18, default OFF. When `True`, replaces `mean_heads` with a single
                lower-triangular-masked `(w_max, w_max)` weight + `(w_max,)` bias (see class
                docstring) instead of the `w_max`-entry `ModuleList`. Each row is initialised from
                an independently-constructed `Linear(w_max, 1)` -- the SAME per-row RNG draw order
                the unfused `mean_heads` loop uses -- so a `fused_heads=True` net built under the
                same seed as a `fused_heads=False` net has IDENTICAL live (unmasked) weights and
                biases; only each row's DEAD tail (columns >= that row's own width, read by neither
                mode) differs, since the fused tail is hard-zeroed at init rather than left at its
                unused random draw.
        """
        super().__init__()
        self.w_max = int(w_max)
        self.trunk = nn.Linear(1, self.w_max)
        self.activation = activation()
        self.fused_heads = bool(fused_heads)
        if self.fused_heads:
            row_weights, row_biases = [], []
            for _ in range(self.w_max):
                head = nn.Linear(self.w_max, 1)
                row_weights.append(head.weight.detach())
                row_biases.append(head.bias.detach())
            mask = torch.tril(torch.ones(self.w_max, self.w_max))  # row k-1 valid at cols <= k-1
            self.register_buffer("_fused_head_mask", mask)
            self.fused_mean_weight = nn.Parameter(torch.cat(row_weights, dim=0) * mask)
            self.fused_mean_bias = nn.Parameter(torch.cat(row_biases, dim=0))
            self.mean_heads = None
        else:
            self.mean_heads = nn.ModuleList(nn.Linear(self.w_max, 1) for _ in range(self.w_max))

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, 1) -> (N, w_max)` post-activation hidden representation (identical to `NestedWidthNet.hidden`)."""
        return self.activation(self.trunk(x))

    def _fused_weight(self) -> torch.Tensor:
        """Effective fused mean-head weight `(w_max, w_max)`, fused mode only.

        The mask buffer is re-applied on every call, so the hard-masked upper triangle (`col > row`)
        is EXACTLY zero in whatever this returns regardless of what an optimizer step does to the
        raw parameter's masked entries (`tests/test_fused_heads_equivalence.py` pins this after real
        Adam steps, not merely at init).
        """
        return self.fused_mean_weight * self._fused_head_mask

    def forward_width(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at one FIXED width `k` (1..w_max) via width-k's OWN mean head, each `(N, 1)`.

        Unfused: prefix-masks `h[:, k:] = 0` before the readout -- the same masking
        `NestedWidthNet.forward_width` uses -- then reads `mean_head_k` off the masked hidden vector.
        Fused: reads row `k-1` of the fused weight (already zero past column `k-1`, see
        `_fused_weight`) against the UNMASKED hidden vector -- the weight's own mask makes columns
        `>= k` of `h` contribute exactly 0 either way, so no separate `h` masking is needed. `log_var`
        is a dummy zero tensor either way (MSE-only; never in this loss's autograd graph, see class
        docstring).
        """
        if not (1 <= k <= self.w_max):
            raise ValueError(f"k={k} out of range [1, {self.w_max}]")
        h = self.hidden(x)
        if self.fused_heads:
            row = self._fused_weight()[k - 1]
            mean = h @ row.unsqueeze(1) + self.fused_mean_bias[k - 1]
            return mean, torch.zeros_like(mean)
        mask = torch.zeros_like(h)
        mask[:, :k] = 1.0
        h_masked = h * mask
        mean = self.mean_heads[k - 1](h_masked)
        return mean, torch.zeros_like(mean)

    def all_widths_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at every width `k=1..w_max`, each shape `(N, w_max)`; column k-1 is width-k's own head.

        Unfused: no cumsum trick (unlike `NestedWidthNet`) -- each width's head is an independent
        parameter set, not a linear function of a shared readout -- so this is `w_max` masked-
        `Linear(w_max -> 1)` matmuls, one per width (same loop shape as
        `IndependentWidthNet.all_widths_forward`). Fused (WSEL-18): the whole `(N, w_max)` table is
        ONE matmul, `h @ fused_weight.T + fused_bias` -- the per-row mask makes row k-1's dot product
        read only h's first k columns, the vectorisation this task exists for.
        """
        if self.fused_heads:
            h = self.hidden(x)
            mean_all = h @ self._fused_weight().t() + self.fused_mean_bias
            return mean_all, torch.zeros_like(mean_all)
        means, logvars = [], []
        for k in range(1, self.w_max + 1):
            mean_k, logvar_k = self.forward_width(x, k)
            means.append(mean_k)
            logvars.append(logvar_k)
        return torch.cat(means, dim=1), torch.cat(logvars, dim=1)

    def sampled_widths_forward(self, x: torch.Tensor, widths: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` for exactly `widths` (order preserved, repeats allowed), trunk evaluated ONCE.

        `capacity_programme` WSEL-12. Unfused: this class's `all_widths_forward` has NO cumsum
        shortcut (each width reads its OWN head, not a linear function of one shared readout) -- it
        loops `forward_width` over every `k` in `1..w_max`, and `forward_width` itself calls
        `self.hidden(x)` fresh each call, so `all_widths_forward` here still recomputes the trunk
        `w_max` TIMES, not once. Gathering columns out of it would not fix the k-dropout training-loop
        defect this task targets (for a sampled subset smaller than `w_max` it would recompute the
        trunk MORE times than the old per-sampled-width loop, not fewer). This method computes `h`
        ONCE and applies only the SAMPLED widths' own heads to it -- the actual fix. Fused (WSEL-18):
        same one-trunk-eval-then-gather shape, but the per-width gather is an `index_select` over the
        fused weight's ROWS (repeats allowed, same as the unfused loop over `widths`) instead of
        `len(widths)` separate module calls.
        """
        h = self.hidden(x)
        for k in widths:
            if not (1 <= k <= self.w_max):
                raise ValueError(f"k={k} out of range [1, {self.w_max}]")
        if self.fused_heads:
            idx = torch.as_tensor([k - 1 for k in widths], dtype=torch.long, device=x.device)
            weight_rows = self._fused_weight().index_select(0, idx)
            bias_rows = self.fused_mean_bias.index_select(0, idx)
            mean = h @ weight_rows.t() + bias_rows
            return mean, torch.zeros_like(mean)
        means, logvars = [], []
        for k in widths:
            mask = torch.zeros_like(h)
            mask[:, :k] = 1.0
            mean = self.mean_heads[k - 1](h * mask)
            means.append(mean)
            logvars.append(torch.zeros_like(mean))
        return torch.cat(means, dim=1), torch.cat(logvars, dim=1)


class SharedReadoutPerWidthAffineNet(nn.Module):
    """Minimum-seam arm: NestedWidthNet's SHARED readout plus a 2-parameter per-width affine.

    Width k's prediction is `a_k * mean_head(h_masked) + c_k` with the ONE shared `mean_head`
    of `NestedWidthNet` and per-width scalars `a_k` (init 1) / `c_k` (init 0) — 2 params per
    width vs w_max+1 for `SharedTrunkPerWidthHeadNet`. Pins WHERE between 0 and w_max+1
    per-width parameters the readout interference (width-MSE verdict §3) is resolved.
    MSE-only: `log_var` is a dummy zero tensor (never in the loss graph).
    """

    def __init__(self, w_max: int = W_MAX_DEFAULT, activation: type[nn.Module] = nn.Tanh) -> None:
        """Builds the shared trunk, ONE shared mean head, and the per-width affine scalars."""
        super().__init__()
        self.w_max = int(w_max)
        self.trunk = nn.Linear(1, self.w_max)
        self.activation = activation()
        self.mean_head = nn.Linear(self.w_max, 1)
        self.affine_scale = nn.Parameter(torch.ones(self.w_max))
        self.affine_bias = nn.Parameter(torch.zeros(self.w_max))

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, 1) -> (N, w_max)` post-activation hidden representation (as `NestedWidthNet.hidden`)."""
        return self.activation(self.trunk(x))

    def forward_width(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at width k: shared readout of the masked hidden, then width-k's affine."""
        if not (1 <= k <= self.w_max):
            raise ValueError(f"k={k} out of range [1, {self.w_max}]")
        h = self.hidden(x)
        mask = torch.zeros_like(h)
        mask[:, :k] = 1.0
        mean = self.affine_scale[k - 1] * self.mean_head(h * mask) + self.affine_bias[k - 1]
        return mean, torch.zeros_like(mean)

    def all_widths_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at every width, each `(N, w_max)`; cumsum trick + per-width affine.

        The shared readout is linear, so the masked-prefix table is the same cumsum as
        `NestedWidthNet.all_widths_forward`; the affine then applies column-wise.
        """
        h = self.hidden(x)
        contrib = h * self.mean_head.weight.squeeze(0)
        mean_all = torch.cumsum(contrib, dim=1) + self.mean_head.bias
        mean_all = mean_all * self.affine_scale + self.affine_bias
        return mean_all, torch.zeros_like(mean_all)

    def sampled_widths_forward(self, x: torch.Tensor, widths: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` for exactly `widths`, trunk evaluated ONCE.

        `capacity_programme` WSEL-12: see `NestedWidthNet.sampled_widths_forward` -- uses the SAME
        explicit-masking formula as `forward_width`, not the `all_widths_forward` cumsum shortcut (only
        algebraically, not bit-for-bit, equivalent to it), so the k-dropout training loop's bit-for-bit
        equivalence bar holds for this class too.
        """
        h = self.hidden(x)
        means = []
        for k in widths:
            if not (1 <= k <= self.w_max):
                raise ValueError(f"k={k} out of range [1, {self.w_max}]")
            mask = torch.zeros_like(h)
            mask[:, :k] = 1.0
            means.append(self.affine_scale[k - 1] * self.mean_head(h * mask) + self.affine_bias[k - 1])
        mean_all = torch.cat(means, dim=1)
        return mean_all, torch.zeros_like(mean_all)


# ---------------------------------------------------------------------------
# executed_flops registration (capacity-programme Task FP-2, completion criterion carried from
# FP-1) -- moved here from `automl_package/examples/capacity_accounting.py`'s shim now that the
# four classes live alongside the dispatcher's import, not across the package/examples boundary.
# Formulas unchanged from before the move.
# ---------------------------------------------------------------------------


@executed_flops.register(NestedWidthNet)
def _executed_flops_nested_width(net: NestedWidthNet, config: int) -> int:
    """Routed width-`config` MACs for `NestedWidthNet` (shared trunk + ONE shared mean head).

    The prefix property (selftests (a)/(b) in the pre-move `nested_width_net.py`) guarantees
    hidden nodes `>= config` never influence a width-`config` output -- they are zeroed before
    both readouts in `forward_width`, so an efficient width-`config` DEPLOYMENT needs only the
    first `config` trunk output rows and the first `config` input columns of `mean_head`, NOT the
    full `w_max`-wide matmuls `forward_width`/`all_widths_forward` literally compute (they compute
    the full width for masking-consistency, not for efficiency). `logvar_head` is excluded
    unconditionally -- see module docstring (MASTER Decision 2).
    """
    k = config
    if not (1 <= k <= net.w_max):
        raise ValueError(f"config={k} out of range [1, {net.w_max}]")
    return _linear_macs(net.trunk.in_features, k) + _linear_macs(k, net.mean_head.out_features)


@executed_flops.register(SharedTrunkPerWidthHeadNet)
def _executed_flops_shared_trunk_per_width_head(net: SharedTrunkPerWidthHeadNet, config: int) -> int:
    """Routed width-`config` MACs for `SharedTrunkPerWidthHeadNet` (shared trunk, per-width heads).

    Same trunk-slicing argument as `NestedWidthNet` (the prefix property is trunk-level, not
    readout-level, so it holds for any head reading the masked hidden vector), then width-
    `config`'s OWN readout, sliced to its own first `config` input columns by the same
    masked-to-zero argument. The readout always produces a single scalar mean output -- true of
    both the unfused `mean_heads[config - 1]` (`Linear(w_max, 1)`) and the WSEL-18 fused row
    (`fused_mean_weight[config - 1]`) -- so this reads that constant directly rather than
    `net.mean_heads[k - 1].out_features`, which is `None` in fused mode (`fused_heads=True`).
    No logvar branch -- this class's `log_var` is a dummy zero tensor, never computed (see class
    docstring).
    """
    k = config
    if not (1 <= k <= net.w_max):
        raise ValueError(f"config={k} out of range [1, {net.w_max}]")
    mean_head_out_features = 1  # both fused and per-head readouts always produce one scalar mean.
    return _linear_macs(net.trunk.in_features, k) + _linear_macs(k, mean_head_out_features)


@executed_flops.register(IndependentWidthNet)
def _executed_flops_independent_width(net: IndependentWidthNet, config: int) -> int:
    """Routed width-`config` MACs for `IndependentWidthNet` (K disjoint sub-nets, no sharing).

    No slicing trick needed here (unlike the shared-trunk classes above) -- sub-net `config` is
    already sized exactly `config`, with literal full compute: `trunk = Linear(1, config)` then
    `mean_head = Linear(config, 1)`. `logvar_head` excluded unconditionally -- module docstring.
    """
    k = config
    if not (1 <= k <= net.w_max):
        raise ValueError(f"config={k} out of range [1, {net.w_max}]")
    sub = net.subnets[k - 1]
    trunk_lin, mean_lin = sub["trunk"], sub["mean_head"]
    return _linear_macs(trunk_lin.in_features, trunk_lin.out_features) + _linear_macs(mean_lin.in_features, mean_lin.out_features)


@executed_flops.register(SharedReadoutPerWidthAffineNet)
def _executed_flops_shared_readout_per_width_affine(net: SharedReadoutPerWidthAffineNet, config: int) -> int:
    """Routed width-`config` MACs for `SharedReadoutPerWidthAffineNet` (shared readout + per-width affine).

    Same trunk- and shared-`mean_head`-slicing argument as `NestedWidthNet`, plus the width-
    `config` affine `a_k * mean + c_k`: one multiply per sample (the shift-add is excluded under
    this module's bias convention, same as every Linear layer here) -- `+1` MAC.
    """
    k = config
    if not (1 <= k <= net.w_max):
        raise ValueError(f"config={k} out of range [1, {net.w_max}]")
    affine_scale_mac = 1
    return _linear_macs(net.trunk.in_features, k) + _linear_macs(k, net.mean_head.out_features) + affine_scale_mac
