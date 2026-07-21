"""Tests for the capacity-programme shared selection primitives (Task FP-9,
`docs/plans/capacity_programme/flexnn-package.md`).

Covers FP-9.a (`cheapest_within_tolerance`, `automl_package/utils/capacity_selection.py`), FP-9.b
(`bootstrap_se`/`two_sample_bootstrap_se`, `automl_package/utils/numerics.py`), and FP-9.c
(`router_fit_cost`/`held_out_read_cost`/`sweep_cost`, `automl_package/utils/capacity_accounting.py`).
Every expected value below is hand-computed in the test, matching this repo's
`test_capacity_accounting.py` convention of named-local expected values rather than bare literal
assertions.
"""

import numpy as np
import pytest

from automl_package.utils.capacity_accounting import (
    DepthNetShapeDescriptor,
    MoEShapeDescriptor,
    executed_flops,
    held_out_read_cost,
    router_fit_cost,
    sweep_cost,
)
from automl_package.utils.capacity_selection import cheapest_within_tolerance
from automl_package.utils.numerics import bootstrap_se, two_sample_bootstrap_se

# ---------------------------------------------------------------------------
# FP-9.b -- bootstrap_se (plain / paired shapes)
# ---------------------------------------------------------------------------


def test_bootstrap_se_matches_sd_over_sqrt_n_within_10_percent():
    """Bootstrap SE of the mean approximates the analytic `sd / sqrt(n)` at large n."""
    rng = np.random.default_rng(0)
    v = rng.normal(0.0, 1.0, size=2000)
    se = bootstrap_se(v, n_boot=2000, seed=0)
    analytic_se = 1.0 / np.sqrt(2000)
    assert abs(se - analytic_se) / analytic_se < 0.10


def test_bootstrap_se_deterministic_under_fixed_seed():
    """Same input, n_boot, and seed must reproduce the same SE (FP-9.b: no silent RNG drift)."""
    rng = np.random.default_rng(0)
    v = rng.normal(0.0, 1.0, size=200)
    assert bootstrap_se(v, n_boot=500, seed=7) == bootstrap_se(v, n_boot=500, seed=7)


def test_bootstrap_se_of_constant_vector_is_exactly_zero():
    """A constant (paired-difference) vector's bootstrap SE is exactly 0.0 -- every resample mean is identical."""
    assert bootstrap_se(np.full(100, 3.0), n_boot=200, seed=0) == 0.0


def test_bootstrap_se_seed_changes_answer():
    """Two different seeds on a non-constant vector should (almost certainly) disagree -- guards against a seed no-op."""
    rng = np.random.default_rng(1)
    v = rng.normal(0.0, 1.0, size=50)
    assert bootstrap_se(v, n_boot=300, seed=0) != bootstrap_se(v, n_boot=300, seed=1)


# ---------------------------------------------------------------------------
# FP-9.b -- two_sample_bootstrap_se
# ---------------------------------------------------------------------------


def test_two_sample_bootstrap_se_matches_analytic_se_of_difference():
    """SE of mean(a) - mean(b) for two independent samples approximates sqrt(se_a^2 + se_b^2)."""
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, size=1000)
    b = rng.normal(0.0, 2.0, size=1000)
    se = two_sample_bootstrap_se(a, b, n_boot=2000, seed=0)
    analytic_se = np.sqrt(1.0**2 / 1000 + 2.0**2 / 1000)
    assert abs(se - analytic_se) / analytic_se < 0.10


def test_two_sample_bootstrap_se_identical_samples_close_to_zero():
    """Two draws from the same distribution: the SE of their difference is small relative to each sample's own spread."""
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 1.0, size=500)
    b = rng.normal(0.0, 1.0, size=500)
    se = two_sample_bootstrap_se(a, b, n_boot=1000, seed=0)
    assert 0.0 < se < 0.5  # sqrt(2/500) ~= 0.063 -- generous bound, just checks it's not degenerate


# ---------------------------------------------------------------------------
# FP-9.a -- cheapest_within_tolerance
# ---------------------------------------------------------------------------


def test_cheapest_within_tolerance_selects_start_of_flat_plateau():
    """A curve flat beyond index 3 must select 3, NOT the argmin (which could land on 3, 4, or 5)."""
    rng = np.random.default_rng(0)
    base = np.array([10.0, 5.0, 2.0, 1.0, 1.0, 1.0])
    errs = base[None, :] + rng.normal(0, 0.01, size=(200, 6))
    assert cheapest_within_tolerance(errs, seed=0) == 3


def test_cheapest_within_tolerance_strictly_decreasing_selects_last():
    """A strictly decreasing curve well outside noise selects the most expensive (last) capacity."""
    rng = np.random.default_rng(0)
    base = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0])
    errs = base[None, :] + rng.normal(0, 0.01, size=(200, 6))
    assert cheapest_within_tolerance(errs, seed=0) == 5


def test_cheapest_within_tolerance_pure_noise_selects_cheapest():
    """A curve with no real per-capacity difference (pure noise) selects the cheapest capacity, index 0."""
    rng = np.random.default_rng(0)
    errs = rng.normal(1.0, 0.05, size=(200, 6))
    assert cheapest_within_tolerance(errs, seed=0) == 0


def test_cheapest_within_tolerance_deterministic_under_fixed_seed():
    """Same table and seed reproduce the same selected index."""
    rng = np.random.default_rng(4)
    errs = np.array([5.0, 3.0, 3.0, 3.0])[None, :] + rng.normal(0, 0.02, size=(150, 4))
    assert cheapest_within_tolerance(errs, seed=1) == cheapest_within_tolerance(errs, seed=1)


def test_cheapest_within_tolerance_single_capacity_selects_it():
    """A one-column table has nothing else to choose -- must select index 0."""
    errs = np.full((50, 1), 2.0)
    assert cheapest_within_tolerance(errs, seed=0) == 0


def test_cheapest_within_tolerance_rejects_non_2d_table():
    with pytest.raises(ValueError, match="2-D"):
        cheapest_within_tolerance(np.zeros(10))


def test_cheapest_within_tolerance_rejects_empty_capacity_grid():
    with pytest.raises(ValueError, match="capacity column"):
        cheapest_within_tolerance(np.zeros((10, 0)))


# ---------------------------------------------------------------------------
# FP-9.c -- selection-cost accounting
# ---------------------------------------------------------------------------


def test_router_fit_cost_matches_hand_computed_mlp_macs():
    """`in_dim=2 -> hidden=(3,) -> n_capacities=4` MLP: forward = 2*3 + 3*4 = 18 MACs/sample."""
    expected_forward_macs = 2 * 3 + 3 * 4
    expected_backward_forward_ratio = 2  # grad-input + grad-weight matmuls, each ~1x forward
    n_samples, n_epochs = 10, 5
    expected = n_epochs * n_samples * expected_forward_macs * (1 + expected_backward_forward_ratio)
    assert router_fit_cost(in_dim=2, n_capacities=4, n_samples=n_samples, n_epochs=n_epochs, hidden=(3,)) == expected


def test_router_fit_cost_requires_hidden():
    """`hidden` has no default -- it must track the router's actual configuration, not a copied constant."""
    with pytest.raises(TypeError):
        router_fit_cost(in_dim=2, n_capacities=4, n_samples=10, n_epochs=5)  # missing `hidden`


def test_held_out_read_cost_sums_forward_only_macs_over_the_grid():
    """`held_out_read_cost` = n_samples * sum of each capacity's own `executed_flops` (no backward)."""
    desc = DepthNetShapeDescriptor(input_size=2, hidden_size=4, output_size=1, max_depth=3)
    # d=1: block0(2,4)=8 + output(4,1)=4 = 12. d=2: 8 + chained(4,4)=16 + 4 = 28. d=3: 8 + 2*16 + 4 = 44.
    expected_flops_d1 = 2 * 4 + 4 * 1
    expected_flops_d2 = 2 * 4 + 4 * 4 + 4 * 1
    expected_flops_d3 = 2 * 4 + 2 * (4 * 4) + 4 * 1
    assert executed_flops(desc, 1) == expected_flops_d1
    assert executed_flops(desc, 2) == expected_flops_d2
    assert executed_flops(desc, 3) == expected_flops_d3
    n_samples = 10
    expected = n_samples * (expected_flops_d1 + expected_flops_d2 + expected_flops_d3)
    assert held_out_read_cost(desc, [1, 2, 3], n_samples=n_samples) == expected


def test_sweep_cost_trains_one_dedicated_model_per_capacity():
    """`sweep_cost` = sum over the grid of `n_epochs * n_train_samples * executed_flops * (1 + backward_ratio)`."""
    moe_desc = MoEShapeDescriptor(d_in=2, expert_hidden=3, n_experts=4)
    # router (2,4): 2*4=8. per_expert: (2,3)=6 + (3,1)=3 = 9. top_k=1: 8+9=17. top_k=2: 8+18=26.
    expected_flops_top1 = 2 * 4 + (2 * 3 + 3 * 1)
    expected_flops_top2 = 2 * 4 + 2 * (2 * 3 + 3 * 1)
    assert executed_flops(moe_desc, 1) == expected_flops_top1
    assert executed_flops(moe_desc, 2) == expected_flops_top2
    n_train_samples, n_epochs, backward_forward_ratio = 3, 2, 2
    expected = n_epochs * n_train_samples * (1 + backward_forward_ratio) * (expected_flops_top1 + expected_flops_top2)
    assert sweep_cost(moe_desc, [1, 2], n_train_samples=n_train_samples, n_epochs=n_epochs) == expected


def test_selection_cost_mechanisms_are_positive_and_ordered_sensibly():
    """Sanity: all three mechanisms return positive costs, and a sweep (N full trainings) costs
    more than a single held-out read (N forward-only scores) over the same grid and sample count."""
    desc = DepthNetShapeDescriptor(input_size=3, hidden_size=5, output_size=1, max_depth=4)
    grid = [1, 2, 3, 4]
    n = 20
    read_cost = held_out_read_cost(desc, grid, n_samples=n)
    sweep = sweep_cost(desc, grid, n_train_samples=n, n_epochs=1)
    fit_cost = router_fit_cost(in_dim=3, n_capacities=len(grid), n_samples=n, n_epochs=1, hidden=(8,))
    assert read_cost > 0
    assert sweep > 0
    assert fit_cost > 0
    assert sweep > read_cost
