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

**WSEL-16's two additions (`docs/plans/capacity_programme/width.md` ~1553-1587, ~1598-1600; §3.7's
RESOLUTION UPDATE 2026-07-22 assigns the tier-2 objective to this task).** `MonotoneGateWidthNet` is
the `A_GATES` stage-1 arm -- the ONLY new architecture WSEL-16 adds (its other four arms are package
classes or, for `A_STOPGRAD`, a loss-shape change on `NestedWidthNet` with NO new class -- see
`automl_package/examples/width_wsel16.py`'s module docstring). `weighted_squared_error` is the
fixed-sigma weighted objective §3.7 requires on tier 2/3 (`(pred - y)^2 / sigma_true(x)^2`, sigma read
from the toy generator's `region` output, never learned) -- implemented ONCE here per §3.7/§3.9 and
imported by every driver that needs it.
"""

from __future__ import annotations

import enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from automl_package.models.architectures.nested_width_net import W_MAX_DEFAULT, NestedWidthNet, SharedTrunkPerWidthHeadNet
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

        The affine is applied BEFORE the mask, and the mask is applied LAST, unconditionally -- this
        order is load-bearing, not stylistic. `gamma_k * 0 == 0` so the scale alone would be harmless
        either way, but `+ beta_k` is not: applying the affine after masking would ADD `beta_k` into
        every column `>= k`, un-zeroing exactly the columns `width.md:532` requires to "provably cannot
        influence the output" -- reviving `mean_heads[k-1]`'s weights there the instant `beta` moves off
        its zero init. `tests/test_prefix_norm_equivalence.py::
        test_affine_beyond_k_columns_stay_zero_with_nonzero_beta` pins this with a non-zero beta (a
        zero-init beta would pass trivially either way).

        Division still happens on the UNMASKED `h` (`h / r_k`, not `(h * mask) / r_k`): `_running_totals_
        r`/`_naive_r_k` both add `PREFIX_NORM_EPS` under the sqrt, so `r_k` is never zero and there is no
        `0 / r_k` NaN risk to guard against by masking first -- masking first only reads as "safer" until
        you add an affine after it, at which point it is the bug above.
        """
        h_norm = h / r_k
        if self.mode is PrefixNormMode.AFFINE:
            h_norm = self.gamma[k - 1] * h_norm + self.beta[k - 1]
        mask = torch.zeros_like(h)
        mask[:, :k] = 1.0
        return h_norm * mask

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


# ---------------------------------------------------------------------------
# WSEL-16 `A_GATES` -- a thin monotone-decay wrapper over `NestedWidthNet` (`width.md` ~1579-1587).
# The ONLY new architecture this task adds (§3.9's reuse inventory); everything else it needs is
# either a package class or, for `A_STOPGRAD`, a loss-shape change with no new class at all.
# ---------------------------------------------------------------------------


class MonotoneGateWidthNet(nn.Module):
    """`A_GATES`: `NestedWidthNet` plus ONE learnable scalar that monotonically decays each unit's contribution.

    Same trunk and the SAME shared `mean_head` as `NestedWidthNet` (composition, `self.base = ...`,
    never a copy -- SS3.9). The only change is the per-unit contribution feeding the width-k running
    sum: `c_j = g_j * w_j * h_j` instead of `NestedWidthNet`'s plain `w_j * h_j`, with
    `g_j = exp(-softplus(nu) * (j - 1))` a single learnable scalar `nu`, initialised so `g_{w_max} = 0.5`
    (`softplus(nu) = ln(2) / (w_max - 1)`, the `w_max=12` case reproducing the spec's literal `g_12 =
    0.5`). `g_j` is monotonically DECREASING in `j` by construction (a `softplus` keeps its rate
    positive) -- no penalty term is added to the loss for this (the strand's no-arbitrary-penalty rule,
    `docs/plans/capacity_programme/width.md` §1); training uses `A_JOINT`'s plain summed loss.

    ⚠️ **This is OUR SIMPLIFICATION of the published monotone-gate mechanism, which derives its gate
    from a variational bound -- we are not reproducing that derivation, only borrowing the idea that a
    monotone-decaying per-unit gate might resolve `NestedWidthNet`'s tug-of-war (`shared/
    width_transformer_port.md` §1). Label it as a simplification wherever this arm is reported.**

    MSE-only, like every class in this module and its base (`log_var` is a dummy zero, never in the
    loss graph, per `width.md` §3.7).
    """

    def __init__(self, w_max: int = W_MAX_DEFAULT, activation: type[nn.Module] = nn.Tanh) -> None:
        """Builds the wrapped `NestedWidthNet` plus the single learnable gate-rate scalar `nu`.

        Args:
            w_max: maximum hidden width, forwarded to the wrapped net unchanged.
            activation: hidden-layer nonlinearity class, forwarded to the wrapped net unchanged.
        """
        super().__init__()
        self.base = NestedWidthNet(w_max=w_max, activation=activation)
        self.w_max = self.base.w_max
        # softplus(nu) = ln(2) / (w_max - 1) => g_{w_max} = exp(-softplus(nu) * (w_max - 1)) = 0.5.
        # Inverted via softplus^{-1}(y) = log(expm1(y)) (nu can be negative; expm1 keeps this stable
        # near y=0, which a w_max large enough to make ln(2)/(w_max-1) small would otherwise erode).
        target_softplus = math.log(2.0) / (self.w_max - 1)
        init_nu = math.log(math.expm1(target_softplus))
        self.nu = nn.Parameter(torch.tensor(init_nu, dtype=torch.float32))

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, 1) -> (N, w_max)` post-activation hidden representation (the wrapped net's, unchanged)."""
        return self.base.hidden(x)

    def _gates(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """`(w_max,)` monotone-decreasing gate `g_j = exp(-softplus(nu) * (j - 1))`, `j = 1..w_max`."""
        j_minus_one = torch.arange(self.w_max, dtype=dtype, device=device)  # 0-indexed (j-1) for j=1..w_max
        return torch.exp(-nnf.softplus(self.nu) * j_minus_one)

    def _gated_contrib(self, h: torch.Tensor) -> torch.Tensor:
        """`(N, w_max)` gated per-unit contribution `c_j = g_j * w_j * h_j` (pre-prefix-sum/mask)."""
        return h * self.base.mean_head.weight.squeeze(0) * self._gates(h.dtype, h.device)

    def forward_width(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at one FIXED width `k` (1..w_max): `b + sum_{j<=k} g_j * w_j * h_j`."""
        if not (1 <= k <= self.w_max):
            raise ValueError(f"k={k} out of range [1, {self.w_max}]")
        h = self.hidden(x)
        mask = torch.zeros_like(h)
        mask[:, :k] = 1.0
        mean = (self._gated_contrib(h) * mask).sum(dim=1, keepdim=True) + self.base.mean_head.bias
        return mean, torch.zeros_like(mean)

    def all_widths_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` at every width `k=1..w_max` in ONE pass; column `k-1` is width-k's readout.

        Same cumsum trick as `NestedWidthNet.all_widths_forward`, applied to the GATED contribution
        instead of the plain one.
        """
        h = self.hidden(x)
        mean_all = torch.cumsum(self._gated_contrib(h), dim=1) + self.base.mean_head.bias
        return mean_all, torch.zeros_like(mean_all)

    def sampled_widths_forward(self, x: torch.Tensor, widths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, log_var)` for exactly `widths` (order preserved, repeats allowed), trunk+gate evaluated ONCE.

        Uses the SAME explicit-masking formula as `forward_width` (not the cumsum shortcut), matching
        `NestedWidthNet.sampled_widths_forward`'s bit-for-bit-equivalence reasoning for the k-dropout
        training loop.
        """
        h = self.hidden(x)
        contrib = self._gated_contrib(h)
        means = []
        for k in widths:
            if not (1 <= k <= self.w_max):
                raise ValueError(f"k={k} out of range [1, {self.w_max}]")
            mask = torch.zeros_like(h)
            mask[:, :k] = 1.0
            means.append((contrib * mask).sum(dim=1, keepdim=True) + self.base.mean_head.bias)
        mean_all = torch.cat(means, dim=1)
        return mean_all, torch.zeros_like(mean_all)


@executed_flops.register(MonotoneGateWidthNet)
def _executed_flops_monotone_gate(net: MonotoneGateWidthNet, config: int) -> int:
    """Routed width-`config` MACs, reused from the wrapped `NestedWidthNet`'s own registered formula.

    The gate multiply is elementwise (one multiply per unit, not a matmul), so it is excluded under
    this module's MAC convention -- the same treatment `_executed_flops_prefix_norm` gives the RMS
    normalisation above.
    """
    return executed_flops(net.base, config)


# ---------------------------------------------------------------------------
# WSEL-16 tier-2/3 objective -- the fixed-sigma weighted squared error (`width.md` §3.7, assigned to
# this task by §3.8's RESOLUTION UPDATE 2026-07-22). Implemented ONCE; every driver imports it.
# ---------------------------------------------------------------------------


def weighted_squared_error(pred: torch.Tensor, y: torch.Tensor, sigma_true: torch.Tensor) -> torch.Tensor:
    """Fixed-sigma weighted squared error: `(pred - y)^2 / sigma_true(x)^2`, mean-reduced.

    `sigma_true` is read from the toy generator's own `region` output (`width.md` §3.7) -- e.g.
    `make_hetero3`'s noisy-easy region uses `HETERO3_NOISY_SIGMA`, its other two regions use
    `HETERO_NOISE_SIGMA` -- NEVER estimated or learned. Up to the constant `1 / (2 * sigma^2)` this is
    the Gaussian log-likelihood with sigma held at the truth, so on `make_hetero`'s single constant
    sigma it is exactly proportional to plain MSE (tier 1 uses plain MSE directly for that reason,
    `width.md` §3.7); on `make_hetero3`'s two-sigma toy it down-weights the noisy region ~100x, which
    plain MSE does not.

    Args:
        pred: `(N,)` model prediction at one width.
        y: `(N,)` target, same scale as `pred`.
        sigma_true: `(N,)` the generator's true per-point noise std, IN THE SAME SCALE as `y`/`pred` --
            if the caller trains on standardized targets (this program's convention, `y_std = (y - my)
            / sy`), `sigma_true` must be `sigma_true_raw / sy`, not the raw generator sigma, since a
            pure linear rescale of `y` rescales its noise std by the same factor.

    Returns:
        Scalar mean weighted squared error.
    """
    return ((pred - y) ** 2 / sigma_true**2).mean()
