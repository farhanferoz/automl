"""The ONE home for candidate width architectures under test.

`docs/plans/capacity_programme/width.md` SS3.9, MASTER Decision 19's boundary rule. Candidates that
pass their gating task get PROMOTED into
`automl_package/models/architectures/nested_width_net.py` by a later task (WSEL-17); until then they
live here, never in a driver and never in a file of their own -- SS3.9's inventory already carries one
duplicate architecture pair and no task may add another.

WSEL-15 (`docs/plans/capacity_programme/width.md` ~1416-1515) is this module's first tenant: the
prefix-RMS-normalisation variants that test whether the certified nested-width design survives a
normalisation layer, the one obstacle standing between this design and a transformer port
(`docs/plans/capacity_programme/shared/width_transformer_port.md` SS5). All three variants below are a
THIN WRAPPER over the certified `SharedTrunkPerWidthHeadNet` -- composition (`self.base = ...`), never
a copy -- reusing its trunk, its activation and its per-width `mean_heads` unchanged; the only thing
this module adds is what happens to the hidden vector between the trunk and each head.

**The exact formula (no discretion, `width.md` ~1445-1463).** For width `k`, root-mean-square
normalisation with NO mean subtraction:

    r_k(x) = sqrt( (1/k) * sum_{j<=k} h_j(x)^2 + eps ),  eps = PREFIX_NORM_EPS = 1e-5

the head then reads `h_j / r_k` for `j <= k` and `0` beyond -- divided by `k`, the ACTIVE count, never
by `w_max`. `PrefixNormMode.RUNNING_TOTALS` (arm B) computes `r_k` for every `k` at once from
`cumsum(h**2)`, one pass covering every width. `PrefixNormMode.NAIVE` (arm D) computes the identical
formula the textbook way -- loop over `k`, slice `h[:, :k]`, take its RMS directly -- and exists ONLY
as the correctness oracle Step 1 checks arm B against
(`tests/test_prefix_norm_equivalence.py`); it is never trained in the WSEL-15 grid.
`PrefixNormMode.AFFINE` (arm C) adds a per-width SCALAR `gamma_k` (init 1) / `beta_k` (init 0) on top
of arm B's normalised vector -- 2 parameters per width, not per unit (a per-unit affine would test
channel recalibration, a different, explicitly out-of-scope question).

Mean-centred (mean-subtracting) normalisation is explicitly OUT of scope for WSEL-15 -- it needs a
second cumulative sum and interacts with the head's bias term, moving two things at once; see the
module docstring of `automl_package/examples/width_wsel15.py`.
"""

from __future__ import annotations

import enum

import torch
import torch.nn as nn

from automl_package.models.architectures.nested_width_net import W_MAX_DEFAULT, SharedTrunkPerWidthHeadNet
from automl_package.utils.capacity_accounting import executed_flops

PREFIX_NORM_EPS = 1e-5  # WSEL-15 spec, no discretion: eps inside the sqrt.


class PrefixNormMode(enum.Enum):
    """Closed set of prefix-normalisation variants under test (`width.md` ~1445-1463; WSEL-15 arms B/C/D)."""

    RUNNING_TOTALS = "running_totals"  # arm B: r_k from ONE cumsum(h**2) pass, no affine.
    AFFINE = "affine"  # arm C: arm B's r_k, plus a per-width SCALAR gamma_k/beta_k (2 params/width).
    NAIVE = "naive"  # arm D: identical formula, looped + sliced -- the correctness oracle, never in the grid.


class PrefixNormWidthNet(nn.Module):
    """Thin wrapper over `SharedTrunkPerWidthHeadNet`: prefix RMS-normalises the hidden vector.

    Applies between the shared trunk and the per-width heads (WSEL-15 arms B/C/D -- see module
    docstring for the formula). Holds a `SharedTrunkPerWidthHeadNet` instance (`self.base`) and reuses
    its trunk, activation and `mean_heads` UNCHANGED; normalisation only changes what those heads read.

    Exposes the same `w_max` / `forward_width` / `all_widths_forward` / `sampled_widths_forward`
    interface as `SharedTrunkPerWidthHeadNet` (and every other width-net class in this program), so it
    is a drop-in for `kdropout_converged_width_experiment._train_kdropout_to_convergence` and
    `nested_width_net._width_mse` unchanged.
    """

    def __init__(self, w_max: int = W_MAX_DEFAULT, activation: type[nn.Module] = nn.Tanh, mode: PrefixNormMode = PrefixNormMode.RUNNING_TOTALS) -> None:
        """Builds the wrapped `SharedTrunkPerWidthHeadNet` plus, for `mode=AFFINE` only, the per-width scalars.

        Args:
            w_max: maximum hidden width, forwarded to the wrapped net unchanged.
            activation: hidden-layer nonlinearity class, forwarded to the wrapped net unchanged.
            mode: which prefix-normalisation variant this instance computes (arm B/C/D).
        """
        super().__init__()
        self.base = SharedTrunkPerWidthHeadNet(w_max=w_max, activation=activation)
        self.w_max = self.base.w_max
        self.mode = mode
        if mode is PrefixNormMode.AFFINE:
            self.gamma = nn.Parameter(torch.ones(self.w_max))
            self.beta = nn.Parameter(torch.zeros(self.w_max))

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, 1) -> (N, w_max)` post-activation hidden representation (the wrapped net's, unchanged)."""
        return self.base.hidden(x)

    def _running_totals_r(self, h: torch.Tensor) -> torch.Tensor:
        """`r_k(x)` for every `k=1..w_max` at once, `(N, w_max)`, from ONE `cumsum(h**2)` pass (arm B).

        Column `k-1` is `r_k`: `cumsum(h*h, dim=1)[:, k-1] = sum_{j<=k} h_j**2` (a prefix sum needs no
        masking -- indices `>= k` simply have not been added into column `k-1` yet), divided by the
        ACTIVE count `k` (never `w_max`), per the exact spec.
        """
        counts = torch.arange(1, self.w_max + 1, dtype=h.dtype, device=h.device)
        return torch.sqrt(torch.cumsum(h * h, dim=1) / counts + PREFIX_NORM_EPS)

    def _naive_r_k(self, h: torch.Tensor, k: int) -> torch.Tensor:
        """`r_k(x)` for ONE `k`, `(N, 1)`, computed the textbook way: slice `h[:, :k]`, take its RMS directly.

        The correctness oracle (arm D) `tests/test_prefix_norm_equivalence.py` checks `_running_totals_r`
        against -- identical formula, deliberately NOT vectorised, so a bug in the cumsum trick shows up
        as a disagreement rather than being baked into both sides.
        """
        h_slice = h[:, :k]
        return torch.sqrt((h_slice * h_slice).mean(dim=1, keepdim=True) + PREFIX_NORM_EPS)

    def _normalized_masked(self, h: torch.Tensor, k: int, r_k: torch.Tensor) -> torch.Tensor:
        """`h_j / r_k` for `j <= k`, `0` beyond -- plus arm C's per-width scalar affine, if `mode=AFFINE`.

        `(h / r_k) * mask` rather than `(h * mask) / r_k`: division and the 0/1 mask commute (`0 / r_k
        == 0`), so the two are algebraically identical; this order avoids a spurious `0/r_k` NaN risk if
        `r_k` were ever exactly zero for a masked-out column (it never is here, since `_running_totals_r`
        and `_naive_r_k` both add `PREFIX_NORM_EPS` under the sqrt, but the safer order costs nothing).
        """
        mask = torch.zeros_like(h)
        mask[:, :k] = 1.0
        h_norm = (h / r_k) * mask
        if self.mode is PrefixNormMode.AFFINE:
            h_norm = self.gamma[k - 1] * h_norm + self.beta[k - 1]
        return h_norm

    def forward_width(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at one FIXED width `k` (1..w_max) via width-k's own (wrapped) mean head.

        `log_var` is a dummy zero, matching `SharedTrunkPerWidthHeadNet` (MSE-only, `width.md` SS3.7).
        """
        if not (1 <= k <= self.w_max):
            raise ValueError(f"k={k} out of range [1, {self.w_max}]")
        h = self.hidden(x)
        r_k = self._naive_r_k(h, k) if self.mode is PrefixNormMode.NAIVE else self._running_totals_r(h)[:, k - 1 : k]
        h_norm = self._normalized_masked(h, k, r_k)
        mean = self.base.mean_heads[k - 1](h_norm)
        return mean, torch.zeros_like(mean)

    def all_widths_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at every width `k=1..w_max`, each `(N, w_max)`; column `k-1` is width-k's own head.

        The trunk and (for non-`NAIVE` modes) the running-totals table are each computed ONCE, reused
        across every `k` -- only the `w_max` head matmuls themselves are looped, same shape as
        `SharedTrunkPerWidthHeadNet.all_widths_forward`.
        """
        h = self.hidden(x)
        r = None if self.mode is PrefixNormMode.NAIVE else self._running_totals_r(h)
        means = []
        for k in range(1, self.w_max + 1):
            r_k = self._naive_r_k(h, k) if r is None else r[:, k - 1 : k]
            means.append(self.base.mean_heads[k - 1](self._normalized_masked(h, k, r_k)))
        mean_all = torch.cat(means, dim=1)
        return mean_all, torch.zeros_like(mean_all)

    def sampled_widths_forward(self, x: torch.Tensor, widths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` for exactly `widths` (order preserved, repeats allowed), trunk evaluated ONCE.

        WSEL-12's k-dropout training-loop contract (`kdropout_converged_width_experiment.
        _sampled_widths_total_loss` dispatches here whenever a net exposes this method): the shared
        trunk AND (for non-`NAIVE` modes) the running-totals table are each computed ONCE regardless of
        `len(widths)`, not once per sampled width.
        """
        h = self.hidden(x)
        r = None if self.mode is PrefixNormMode.NAIVE else self._running_totals_r(h)
        means = []
        for k in widths:
            if not (1 <= k <= self.w_max):
                raise ValueError(f"k={k} out of range [1, {self.w_max}]")
            r_k = self._naive_r_k(h, k) if r is None else r[:, k - 1 : k]
            means.append(self.base.mean_heads[k - 1](self._normalized_masked(h, k, r_k)))
        mean_all = torch.cat(means, dim=1)
        return mean_all, torch.zeros_like(mean_all)


@executed_flops.register(PrefixNormWidthNet)
def _executed_flops_prefix_norm(net: PrefixNormWidthNet, config: int) -> int:
    """Routed width-`config` MACs, reused from the wrapped net's own registered formula.

    Identical to the wrapped `SharedTrunkPerWidthHeadNet`'s own registered formula (trunk slice +
    width-k's own head) -- reused via `executed_flops(net.base, k)`, not re-derived. The RMS
    normalisation (a division) and arm C's scalar affine (a multiply-add) are elementwise, not matmuls,
    so this module's MAC convention (`_linear_macs`, matmuls only -- see `capacity_accounting.py`'s
    module docstring) counts none of it, same as it never counts bias adds.
    """
    return executed_flops(net.base, config)
