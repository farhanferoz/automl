"""DistilledCapacityRouter -- Decision-13 selection as a package API (capacity-programme Task F3).

Post-hoc, never-in-training per-input capacity selection: MASTER Decision 13
(`docs/plans/capacity_programme/MASTER.md`) is that capacity routers -- width, depth, joint,
transformer -- are DISTILLED from a held-out per-capacity error table, never learned jointly with
the underlying model (in-training selection strategies such as GumbelSoftmax/SoftGating/STE/
REINFORCE stay exactly as they are; this is an alternative, post-hoc path). One router class
serves every 1-D capacity axis in this programme -- FlexNN depth
(`flexible_neural_network.py::FlexibleHiddenLayersNN.fit_router`), `FlexibleWidthNN` width
(`flexible_width_network.py::FlexibleWidthNN.fit_router`), and (task F9) ProbReg's k -- via a
`capacity_grid: list[tuple[int, ...]]` representation that also leaves room for a future 2-D
(width, depth) joint grid without any joint-specific code here (an explicit F3 non-goal).

Labeling rule (COPIED, not imported, with provenance cited): the cheapest capacity within
`tolerance` of each input's best-achieving capacity, matching
`sinc_width_experiment._cheapest_within_tolerance_labels`
(`automl_package/examples/sinc_width_experiment.py:414-423`) and its `DELTA_TIE = 0.25`
(`sinc_width_experiment.py:333`). Not imported because that module is a research example script
(a leading-underscore private helper); package code under `models/` should not depend on
`examples/`.

Router architecture (COPIED, not imported, with provenance cited): a small
`x (in_dim vector) -> logits over len(capacity_grid)` MLP, hidden `(32, 32)` + ReLU, trained with
Adam + cross-entropy -- the same pattern as `depth_selection_toy._VectorRouterMLP` /
`_train_vector_router` (`automl_package/examples/depth_selection_toy.py:607-639`), itself a
vector-input generalization of `capacity_ladder_k6._RouterMLP`
(`automl_package/examples/capacity_ladder_k6.py:75-89`), which hardcodes a scalar input
(`dims = [1, *hidden]`) and so does not generalize to the vector inputs FlexNN/FlexibleWidthNN use
(see `depth_selection_toy.py:594-603`'s comment on exactly this point).

Cost accounting: the caller supplies `cost_fn(capacity) -> float`. This module has NO FLOPs
default and no dependency on any specific net class or on `examples/` -- each model class wires
its OWN default `cost_fn` from `automl_package/examples/capacity_accounting.py`'s S2
`executed_flops` (imported there, not re-derived; see `FlexibleHiddenLayersNN.fit_router` /
`FlexibleWidthNN.fit_router`), which is the ONLY place FLOPs arithmetic for this programme lives.
Keeping this module net-agnostic is what lets the SAME class drop into a third capacity axis (k,
task F9) with zero changes.

The router never sees an oracle difficulty/regime label, the error table itself at inference
time, or the capacity index as an input feature -- only raw `x` (`fit`'s `eval_fn` closes over
whatever the caller needs to score a capacity; `fit` itself only ever calls
`eval_fn(x_val, capacity)` and reads its returned per-sample error array).

Router reconciliation (capacity-programme Task FP-5): this class is the single implementation
for what it covers. `_CapacityRouterMLP` already generalizes `capacity_ladder_k6._RouterMLP`'s
scalar-only input layer (it infers `in_dim` from `x`, so a scalar `x` -- reshaped to `(N, 1)` --
is just the `in_dim == 1` case, not a separate code path). `fit_soft()` ports
`capacity_ladder_k6._train_router`'s soft-target branch and `capacity_ladder_t2._train_router`
(vector-input soft-label CE) as a second training path alongside `fit()`'s hard-label one --
both now share the same `_CapacityRouterMLP` and the same `_fit_from_targets` training loop.
`blend_scores()`/`blend_nll()` port `capacity_ladder_t2._blend_scores`/`_blend_nll` (the
blend-likelihood evaluation path `fit()`/`route()` alone do not provide).

What is deliberately NOT ported -- experiment-protocol, not library, and it must stay that way
(see `docs/plans/capacity_programme/shared/router-capabilities.md` for the full classification):
soft-target *construction* (`capacity_ladder_k6._soft_targets`'s per-tercile EM-stacked
responsibilities, `capacity_ladder_t2._knn_soft_targets`'s kNN generalization of it) stays with
the drivers -- `fit_soft()` trains on whatever `(N, n_capacities)` distribution the caller
supplies, it does not construct one; the direct-objective trainer
(`capacity_ladder_s2._train_router_direct`) is an unresolved research arm (PROTECTED.tsv);
and the five-arm label-construction factorial (`capacity_ladder_s1.ARM_NAMES`) is a certified
comparison whose four losing arms have no reason to exist outside that comparison.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from automl_package.utils.pytorch_utils import get_device

DEFAULT_TOLERANCE = 0.25  # matches sinc_width_experiment.py:333 DELTA_TIE
DEFAULT_HIDDEN: tuple[int, ...] = (32, 32)  # capacity_ladder_k6.py:64 HIDDEN convention
DEFAULT_N_EPOCHS = 300  # capacity_ladder_k6.py:65 N_EPOCHS
DEFAULT_LR = 1e-2  # capacity_ladder_k6.py:66 LR


def _as_capacity_input(x: np.ndarray) -> np.ndarray:
    """`(N,)` -> `(N, 1)`; `(N, in_dim)` passed through -- the shared scalar/vector reshape.

    The same reshape `fit`/`fit_soft`/`route_index`/`blend_scores` all need, and the reason
    `_CapacityRouterMLP` needs no separate scalar code path: a scalar `x` is just `in_dim == 1`
    after this call (matches `capacity_ladder_t2._as_2d`, generalizing
    `capacity_ladder_k6`'s hardcoded `.reshape(-1, 1)`).
    """
    x_arr = np.asarray(x, dtype=np.float32)
    return x_arr.reshape(-1, 1) if x_arr.ndim == 1 else x_arr


def _cheapest_within_tolerance_labels(error_table: np.ndarray, tolerance: float = DEFAULT_TOLERANCE) -> np.ndarray:
    """Smallest-index capacity with `error <= (1 + tolerance) * row_min`, per row.

    Copied from `sinc_width_experiment._cheapest_within_tolerance_labels`
    (`automl_package/examples/sinc_width_experiment.py:414-423`, see module docstring for why this
    is a copy rather than an import). `argmax` on a boolean row returns the FIRST True -- the
    cheapest (lowest-index) capacity meeting the tolerance -- because `capacity_grid` columns are
    ordered cheapest-first by this module's convention.

    Args:
        error_table: `(N, n_capacities)` per-sample error at each capacity, lower is better.
        tolerance: relative tolerance above the row-wise minimum error.

    Returns:
        `(N,)` 0-based column index per row.
    """
    min_error = error_table.min(axis=1, keepdims=True)
    within_tolerance = error_table <= (1.0 + tolerance) * min_error
    return within_tolerance.argmax(axis=1)


class _CapacityRouterMLP(nn.Module):
    """`x (in_dim vector) -> logits over n_capacities` classifier, hidden `(32, 32)` + ReLU.

    Copied from `depth_selection_toy._VectorRouterMLP`
    (`automl_package/examples/depth_selection_toy.py:607-621`), itself a vector-input
    generalization of `capacity_ladder_k6._RouterMLP` (see module docstring for why that class
    doesn't generalize, and why this is a copy rather than an import).
    """

    def __init__(self, in_dim: int, n_capacities: int, hidden: tuple[int, ...] = DEFAULT_HIDDEN) -> None:
        """Builds the hidden stack plus a final `Linear(-> n_capacities)` logit layer."""
        super().__init__()
        dims = [in_dim, *hidden]
        layers: list[nn.Module] = []
        for d_in, d_out in itertools.pairwise(dims):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], n_capacities))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, in_dim) -> (N, n_capacities)` logits."""
        return self.net(x)


class DistilledCapacityRouter:
    """Decision-13 distilled per-input capacity selector -- see module docstring.

    `fit()` builds a held-out per-capacity error table via a caller-supplied `eval_fn`, labels
    each input with the cheapest capacity within `tolerance` of its best-achieving capacity, and
    trains a small MLP (raw input `x` -> capacity index) on those labels. `route()` then maps new
    inputs to a capacity from `capacity_grid` via that trained router.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = DEFAULT_HIDDEN,
        n_epochs: int = DEFAULT_N_EPOCHS,
        lr: float = DEFAULT_LR,
        seed: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        """Configures the router MLP's hyperparameters; `fit()` builds and trains it.

        Args:
            hidden: router MLP hidden-layer sizes.
            n_epochs: full-batch Adam epochs to train the router for.
            lr: Adam learning rate.
            seed: torch RNG seed for router-weight initialization.
            device: torch device for the router MLP and its inputs. `None` (default)
                auto-detects via `get_device()` (CUDA > XPU > CPU, honoring `AUTOML_DEVICE`).
        """
        self.hidden = hidden
        self.n_epochs = n_epochs
        self.lr = lr
        self.seed = seed
        self.device = device if device is not None else get_device()
        self.capacity_grid: list[tuple[int, ...]] | None = None
        self.costs_: np.ndarray | None = None
        self.router_: _CapacityRouterMLP | None = None

    def fit(
        self,
        eval_fn: Callable[[np.ndarray, tuple[int, ...]], np.ndarray],
        x_val: np.ndarray,
        y_val: np.ndarray,
        capacity_grid: list[tuple[int, ...]],
        tolerance: float = DEFAULT_TOLERANCE,
        cost_fn: Callable[[tuple[int, ...]], float] | None = None,
    ) -> DistilledCapacityRouter:
        """Fits the router on a held-out `(x_val, y_val)` set.

        Args:
            eval_fn: `eval_fn(x_val, capacity) -> (N,)` per-sample error at `capacity`, lower is
                better. Supplied by the caller (each model class wires its own, closing over
                whatever it needs from `y_val` to score a prediction -- e.g. squared error for
                regression, 0/1 error for classification, per-sample NLL for a probabilistic
                model). This method never computes error itself and never reads `y_val` beyond
                the length check below, so it is agnostic to whatever loss `eval_fn` uses
                internally.
            x_val: `(N,)` or `(N, in_dim)` held-out inputs.
            y_val: held-out targets, used only to validate `len(y_val) == len(x_val)` here (the
                actual scoring lives inside `eval_fn`).
            capacity_grid: candidate capacities, cheapest first, e.g. `[(4,), (7,), (10,)]` for a
                1-D depth/width grid (a future joint grid would use `[(w, d), ...]` tuples -- the
                tuple representation is chosen so that extension needs no change here).
            tolerance: relative tolerance for the cheapest-within-tolerance labeling rule.
            cost_fn: `cost_fn(capacity) -> float`, used by `mean_deployed_cost`. `None` disables
                `mean_deployed_cost` (a `RuntimeError` if called).

        Returns:
            self.
        """
        if len(x_val) != len(y_val):
            raise ValueError(f"x_val and y_val must have the same length, got {len(x_val)} and {len(y_val)}.")
        if not capacity_grid:
            raise ValueError("capacity_grid must be non-empty.")

        x_arr = _as_capacity_input(x_val)
        error_table = np.stack([np.asarray(eval_fn(x_val, capacity)) for capacity in capacity_grid], axis=1)
        labels = _cheapest_within_tolerance_labels(error_table, tolerance)

        self._fit_from_targets(x_arr, capacity_grid, cost_fn, hard_labels=labels)
        return self

    def fit_soft(
        self,
        x_val: np.ndarray,
        soft_targets: np.ndarray,
        capacity_grid: list[tuple[int, ...]],
        cost_fn: Callable[[tuple[int, ...]], float] | None = None,
    ) -> DistilledCapacityRouter:
        """Fits the router by soft-label cross-entropy to a caller-supplied target distribution.

        The soft-target training path ported from `capacity_ladder_k6._train_router`'s
        `soft_targets` branch and `capacity_ladder_t2._train_router` (see module docstring).
        Unlike `fit()`, this method builds no error table and derives no labels itself -- it
        trains directly on whatever `(N, n_capacities)` distribution the caller supplies.
        Constructing that distribution (e.g. K6's per-tercile EM-stacked responsibilities, T2's
        kNN generalization of it) is experiment-protocol and stays with the caller; see
        `docs/plans/capacity_programme/shared/router-capabilities.md`.

        Args:
            x_val: `(N,)` or `(N, in_dim)` held-out inputs.
            soft_targets: `(N, len(capacity_grid))` per-row target distribution (need not be
                normalized; `nnf.log_softmax` on the router's own logits is what normalizes the
                *prediction* side -- callers typically pass an already-normalized distribution
                for `soft_targets` itself, as K6/T2 do).
            capacity_grid: candidate capacities, cheapest first (see `fit()`).
            cost_fn: `cost_fn(capacity) -> float`, used by `mean_deployed_cost` (see `fit()`).

        Returns:
            self.
        """
        if len(x_val) != len(soft_targets):
            raise ValueError(f"x_val and soft_targets must have the same length, got {len(x_val)} and {len(soft_targets)}.")
        if not capacity_grid:
            raise ValueError("capacity_grid must be non-empty.")
        soft_targets_arr = np.asarray(soft_targets, dtype=np.float32)
        if soft_targets_arr.shape[1] != len(capacity_grid):
            raise ValueError(f"soft_targets has {soft_targets_arr.shape[1]} columns, expected len(capacity_grid) == {len(capacity_grid)}.")

        x_arr = _as_capacity_input(x_val)
        self._fit_from_targets(x_arr, capacity_grid, cost_fn, soft_targets=soft_targets_arr)
        return self

    def _fit_from_targets(
        self,
        x_arr: np.ndarray,
        capacity_grid: list[tuple[int, ...]],
        cost_fn: Callable[[tuple[int, ...]], float] | None,
        *,
        hard_labels: np.ndarray | None = None,
        soft_targets: np.ndarray | None = None,
    ) -> None:
        """Shared training loop behind `fit()` (hard labels) and `fit_soft()` (soft targets).

        Exactly one of `hard_labels`/`soft_targets` is given -- matches
        `capacity_ladder_k6._train_router`'s own mutual-exclusivity contract.
        """
        if (hard_labels is None) == (soft_targets is None):
            raise ValueError("exactly one of hard_labels or soft_targets must be given")

        torch.manual_seed(self.seed)
        self.router_ = _CapacityRouterMLP(x_arr.shape[1], len(capacity_grid), hidden=self.hidden).to(self.device)
        optimizer = torch.optim.Adam(self.router_.parameters(), lr=self.lr)
        x_tensor = torch.as_tensor(x_arr, dtype=torch.float32, device=self.device)
        if hard_labels is not None:
            target_tensor = torch.as_tensor(hard_labels, dtype=torch.long, device=self.device)
        else:
            target_tensor = torch.as_tensor(soft_targets, dtype=torch.float32, device=self.device)

        self.router_.train()
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            logits = self.router_(x_tensor)
            loss = nnf.cross_entropy(logits, target_tensor) if hard_labels is not None else -(target_tensor * nnf.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
        self.router_.eval()

        self.capacity_grid = list(capacity_grid)
        self.costs_ = np.array([cost_fn(capacity) for capacity in capacity_grid], dtype=float) if cost_fn is not None else None

    def route_index(self, x: np.ndarray) -> np.ndarray:
        """Routed capacity-grid index per sample, `(N,)` int array (0-based, into `capacity_grid`)."""
        if self.router_ is None:
            raise RuntimeError("DistilledCapacityRouter.fit() must be called before routing.")
        x_arr = _as_capacity_input(x)
        with torch.no_grad():
            logits = self.router_(torch.as_tensor(x_arr, dtype=torch.float32, device=self.device))
        return logits.argmax(dim=1).cpu().numpy()

    def route(self, x: np.ndarray) -> list[tuple[int, ...]]:
        """Routed capacity per sample: an element of `capacity_grid` for each row of `x`."""
        if self.capacity_grid is None:
            raise RuntimeError("DistilledCapacityRouter.fit() must be called before routing.")
        return [self.capacity_grid[i] for i in self.route_index(x)]

    def blend_scores(self, x: np.ndarray, score: np.ndarray) -> np.ndarray:
        """Per-example blended log-score: `logsumexp_c(log_softmax(router_logits(x))_c + score[:, c])`.

        Ported from `capacity_ladder_t2._blend_scores` (see module docstring) -- the
        blend-likelihood evaluation path `route`/`route_index` do not provide. Combines the
        router's own soft weighting over `capacity_grid` with a caller-supplied
        `(N, n_capacities)` per-capacity log-likelihood table `score`, columns aligned with
        `capacity_grid`.

        Args:
            x: `(N,)` or `(N, in_dim)` inputs, same convention as `route`/`route_index`.
            score: `(N, len(capacity_grid))` per-capacity log-likelihood, lower-index columns
                matching the same `capacity_grid` order `fit`/`fit_soft` were called with.

        Returns:
            `(N,)` blended log-score per row.
        """
        if self.router_ is None:
            raise RuntimeError("DistilledCapacityRouter.fit() or fit_soft() must be called before blend_scores.")
        x_arr = _as_capacity_input(x)
        with torch.no_grad():
            logits = self.router_(torch.as_tensor(x_arr, dtype=torch.float32, device=self.device))
            log_w = nnf.log_softmax(logits, dim=1).double()
        score_t = torch.as_tensor(np.asarray(score, dtype=np.float64), device=self.device)
        return torch.logsumexp(log_w + score_t, dim=1).cpu().numpy()

    def blend_nll(self, x: np.ndarray, score: np.ndarray) -> float:
        """Blended NLL: `-mean(blend_scores(x, score))` -- ports `capacity_ladder_t2._blend_nll`."""
        return float(-self.blend_scores(x, score).mean())

    def mean_deployed_cost(self, x: np.ndarray) -> float:
        """Mean `cost_fn(capacity)` over the routed capacity per sample.

        Raises:
            RuntimeError: `fit()` was not called with a `cost_fn`.
        """
        if self.costs_ is None:
            raise RuntimeError("mean_deployed_cost requires cost_fn to have been passed to fit().")
        return float(self.costs_[self.route_index(x)].mean())
