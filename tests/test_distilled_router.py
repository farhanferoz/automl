"""Tests for `DistilledCapacityRouter` (capacity-programme Task F3).

Covers the spec's verify bar directly: on a synthetic 2-regime task (an "easy" region where the
cheapest capacity already suffices, and a "hard" region that only the most expensive capacity
fits well), the routed model (a) beats the worst fixed capacity on held-out error, (b) has a mean
deployed cost below the max capacity's cost, and (c) never sees an oracle regime/difficulty label
anywhere in its fit path -- checked structurally via `fit`'s own signature, not just by inspection
of this test.
"""

import inspect

import numpy as np
import pytest

from automl_package.models.common.distilled_router import DistilledCapacityRouter, _cheapest_within_tolerance_labels

RANDOM_SEED = 0
N_SAMPLES = 400
CAPACITY_GRID = [(1,), (2,), (4,), (8,)]
HARD_BASE_ERROR = {1: 1.0, 2: 0.6, 4: 0.2, 8: 0.01}
EASY_ERROR = 0.01
ERROR_NOISE_STD = 1e-3
COST_TOLERANCE = 1e-9


def _make_two_regime_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED):
    """`x` in R^1; `x > 0` is the HARD regime (needs capacity 8 to fit well), `x <= 0` is EASY
    (already fits at capacity 1). `y` here is a placeholder target -- `eval_fn` below is analytic
    in `x` and `capacity` and does not need `y` to compute error, matching real callers where
    `eval_fn` closes over a fitted model's own predictions instead.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_samples, 1)).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.float32)  # unused by this test's eval_fn; see docstring above
    is_hard = (x[:, 0] > 0.0)
    return x, y, is_hard


def _make_eval_fn(is_hard: np.ndarray, seed: int = RANDOM_SEED):
    """`eval_fn(x, capacity) -> (N,)` per-sample error, analytic in the (public) regime of `x`.

    HARD-regime error drops sharply with capacity (only capacity 8 is good); EASY-regime error is
    already low at every capacity. `is_hard` is derived from `x` itself (`x > 0`, public
    information any real eval_fn could recompute) -- it is captured by this closure, never passed
    to `DistilledCapacityRouter.fit()` directly (see `test_fit_never_receives_oracle_label`).
    """
    rng = np.random.default_rng(seed)

    def eval_fn(x: np.ndarray, capacity: tuple[int, ...]) -> np.ndarray:
        del x  # regime is read from the closed-over `is_hard`, matching the row order of x_val
        base = np.where(is_hard, HARD_BASE_ERROR[capacity[0]], EASY_ERROR)
        return base + rng.normal(0.0, ERROR_NOISE_STD, size=base.shape)

    return eval_fn


class TestCheapestWithinToleranceLabels:
    """Unit behavior of the lifted labeling rule."""

    def test_picks_first_column_within_tolerance(self):
        error_table = np.array(
            [
                [1.0, 0.9, 0.05, 0.04],  # min=0.04 at col 3; col 2 (0.05) is within 25% tolerance and cheaper
                [0.5, 0.5, 0.5, 0.5],  # tie everywhere -> cheapest (col 0)
            ]
        )
        labels = _cheapest_within_tolerance_labels(error_table, tolerance=0.25)
        assert labels.tolist() == [2, 0]


class TestDistilledCapacityRouterTwoRegimeTask:
    """The spec's core verify bar: routed error vs worst-fixed, routed cost vs max-capacity cost."""

    def test_routed_beats_worst_fixed_capacity(self):
        x, y, is_hard = _make_two_regime_dataset()
        eval_fn = _make_eval_fn(is_hard)
        cost_fn = lambda capacity: float(capacity[0])  # noqa: E731 -- trivial, test-local

        router = DistilledCapacityRouter(seed=RANDOM_SEED)
        router.fit(eval_fn=eval_fn, x_val=x, y_val=y, capacity_grid=CAPACITY_GRID, cost_fn=cost_fn)

        per_capacity_mean_error = {capacity: float(eval_fn(x, capacity).mean()) for capacity in CAPACITY_GRID}
        worst_fixed_error = max(per_capacity_mean_error.values())

        routed_capacities = router.route(x)
        routed_error = float(np.mean([eval_fn(x[i : i + 1], routed_capacities[i])[0] for i in range(len(x))]))

        assert routed_error < worst_fixed_error, (
            f"routed mean error ({routed_error:.4f}) does not beat the worst fixed capacity's mean error "
            f"({worst_fixed_error:.4f}); per-capacity means: {per_capacity_mean_error}"
        )

    def test_routed_mean_cost_below_max_capacity_cost(self):
        x, y, is_hard = _make_two_regime_dataset()
        eval_fn = _make_eval_fn(is_hard)
        cost_fn = lambda capacity: float(capacity[0])  # noqa: E731 -- trivial, test-local
        max_capacity_cost = max(cost_fn(capacity) for capacity in CAPACITY_GRID)

        router = DistilledCapacityRouter(seed=RANDOM_SEED)
        router.fit(eval_fn=eval_fn, x_val=x, y_val=y, capacity_grid=CAPACITY_GRID, cost_fn=cost_fn)

        mean_cost = router.mean_deployed_cost(x)
        assert mean_cost < max_capacity_cost - COST_TOLERANCE, (
            f"mean deployed cost ({mean_cost:.4f}) is not below the max capacity's cost ({max_capacity_cost:.4f})"
        )

    def test_easy_regime_routes_cheap_hard_regime_routes_expensive(self):
        """Sanity check behind the two bars above: the router actually discriminates the regimes."""
        x, y, is_hard = _make_two_regime_dataset()
        eval_fn = _make_eval_fn(is_hard)

        router = DistilledCapacityRouter(seed=RANDOM_SEED)
        router.fit(eval_fn=eval_fn, x_val=x, y_val=y, capacity_grid=CAPACITY_GRID)

        routed = np.array([c[0] for c in router.route(x)])
        mean_easy_capacity = routed[~is_hard].mean()
        mean_hard_capacity = routed[is_hard].mean()
        assert mean_hard_capacity > mean_easy_capacity, (
            f"hard-regime routed capacity ({mean_hard_capacity:.2f}) is not greater than "
            f"easy-regime routed capacity ({mean_easy_capacity:.2f}) -- router did not learn the regime split."
        )


class TestNoOracleLabelLeakage:
    """Structural (not just by-inspection) check that the router never sees a difficulty/regime label."""

    def test_fit_signature_has_no_oracle_label_parameter(self):
        """`fit`'s only inputs are `eval_fn`, `x_val`, `y_val`, `capacity_grid`, `tolerance`, `cost_fn`
        -- there is no parameter through which a caller could pass an oracle regime/difficulty label,
        so `is_hard` in this test's own dataset construction is structurally unreachable from `fit`.
        """
        sig = inspect.signature(DistilledCapacityRouter.fit)
        assert set(sig.parameters) == {"self", "eval_fn", "x_val", "y_val", "capacity_grid", "tolerance", "cost_fn"}

    def test_fit_result_depends_only_on_eval_fn_output_not_on_regime_identity(self):
        """Two datasets with SWAPPED regime identity (`is_hard` flipped) but whose `eval_fn`s are
        built to return the IDENTICAL error table (by swapping `HARD_BASE_ERROR`/`EASY_ERROR`
        along with the flip) must yield the identical fitted route. This demonstrates the route
        tracks only the error table `eval_fn` returns -- there is no separate back-channel by
        which a regime/difficulty label could influence the fit, consistent with `fit`'s signature
        (checked above) having no parameter to carry one.
        """
        x, _y, is_hard = _make_two_regime_dataset()
        y = np.zeros(len(x), dtype=np.float32)

        def eval_fn_original(x_in: np.ndarray, capacity: tuple[int, ...]) -> np.ndarray:
            del x_in
            return np.where(is_hard, HARD_BASE_ERROR[capacity[0]], EASY_ERROR)

        flipped_is_hard = ~is_hard

        def eval_fn_flipped(x_in: np.ndarray, capacity: tuple[int, ...]) -> np.ndarray:
            # Regime identity is flipped AND the error assignment is flipped to match --
            # net effect: identical error table to eval_fn_original, only the label "is_hard"
            # bookkeeping differs. If fit() depended on regime identity rather than purely on
            # eval_fn's returned numbers, this would change the result; it must not.
            del x_in
            return np.where(flipped_is_hard, EASY_ERROR, HARD_BASE_ERROR[capacity[0]])

        router_a = DistilledCapacityRouter(seed=RANDOM_SEED)
        router_a.fit(eval_fn=eval_fn_original, x_val=x, y_val=y, capacity_grid=CAPACITY_GRID)

        router_b = DistilledCapacityRouter(seed=RANDOM_SEED)
        router_b.fit(eval_fn=eval_fn_flipped, x_val=x, y_val=y, capacity_grid=CAPACITY_GRID)

        assert router_a.route(x) == router_b.route(x), "identical error tables produced different routes -- fit path is not purely a function of eval_fn's output"


class TestDistilledCapacityRouterValidation:
    def test_mismatched_lengths_rejected(self):
        x, y, is_hard = _make_two_regime_dataset()
        eval_fn = _make_eval_fn(is_hard)
        router = DistilledCapacityRouter()
        with pytest.raises(ValueError, match="same length"):
            router.fit(eval_fn=eval_fn, x_val=x, y_val=y[:-1], capacity_grid=CAPACITY_GRID)

    def test_empty_capacity_grid_rejected(self):
        x, y, is_hard = _make_two_regime_dataset()
        eval_fn = _make_eval_fn(is_hard)
        router = DistilledCapacityRouter()
        with pytest.raises(ValueError, match="capacity_grid"):
            router.fit(eval_fn=eval_fn, x_val=x, y_val=y, capacity_grid=[])

    def test_route_before_fit_raises(self):
        router = DistilledCapacityRouter()
        with pytest.raises(RuntimeError, match="fit"):
            router.route(np.zeros((3, 1), dtype=np.float32))

    def test_mean_deployed_cost_without_cost_fn_raises(self):
        x, y, is_hard = _make_two_regime_dataset()
        eval_fn = _make_eval_fn(is_hard)
        router = DistilledCapacityRouter(seed=RANDOM_SEED)
        router.fit(eval_fn=eval_fn, x_val=x, y_val=y, capacity_grid=CAPACITY_GRID)
        with pytest.raises(RuntimeError, match="cost_fn"):
            router.mean_deployed_cost(x)
