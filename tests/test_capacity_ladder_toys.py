"""Ground-truth tests for the capacity-ladder input-varying-capacity toys (F1, execution plan
docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md §3 WS2): Toy G (varying need), Toy G-flat
(negative control) and Toy H (SNR dial).

Run DIRECTLY, e.g. ``python3 tests/test_capacity_ladder_toys.py`` (repo-wide pytest collection is
broken by an omegaconf conflict; this file also has a ``__main__`` runner, matching
``tests/test_variational_em.py``).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "automl_package", "examples"))
import _capacity_ladder_toys as cl  # noqa: E402

N_LARGE = 300_000
TOL = 0.01


def test_toy_g_shapes():
    x, y = cl.make_toy_g(n=123, seed=0)
    assert x.shape == (123, 1)
    assert y.shape == (123,)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


def test_toy_g_flat_shapes():
    x, y = cl.make_toy_g_flat(n=123, seed=0)
    assert x.shape == (123, 1)
    assert y.shape == (123,)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


def test_toy_h_shapes():
    x, y = cl.make_toy_h(n=123, seed=0)
    assert x.shape == (123, 1)
    assert y.shape == (123,)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


def test_toy_g_region_matches_sign_of_x():
    x = np.array([-1.0, -0.001, 0.0, 0.001, 1.0])
    region = cl.toy_g_region(x)
    assert list(region) == [0, 0, 1, 1, 1]


def test_toy_g_signal_continuous_at_boundary():
    # both region formulas hit 0 at x = 0 by construction (a*0 = 0; amplitude*sin(0)*1 = 0)
    signal = cl.toy_g_signal(np.array([-1e-6, 0.0, 1e-6]))
    assert abs(float(signal[1])) < 1e-9
    assert abs(float(signal[0]) - float(signal[2])) < 1e-4


def test_toy_g_halves_have_matched_marginal_variance():
    # the F1 registration: matched marginal variance across the two halves, proving Toy G's
    # regions differ only in SHAPE, not in scale.
    x, y = cl.make_toy_g(n=N_LARGE, seed=0)
    region = cl.toy_g_region(x.ravel())
    var_lo = float(np.var(y[region == 0]))
    var_hi = float(np.var(y[region == 1]))
    assert abs(var_lo - var_hi) < TOL, (var_lo, var_hi)
    target = cl.toy_g_region_variance()
    assert abs(var_lo - target) < TOL, (var_lo, target)
    assert abs(var_hi - target) < TOL, (var_hi, target)


def test_toy_g_flat_matches_toy_g_marginal_variance():
    # G-flat's "same marginal statistics as G" property, checked directly against G's own draws.
    xg, yg = cl.make_toy_g(n=N_LARGE, seed=1)
    xf, yf = cl.make_toy_g_flat(n=N_LARGE, seed=1)
    region_g = cl.toy_g_region(xg.ravel())
    region_f = cl.toy_g_region(xf.ravel())
    for half in (0, 1):
        var_g = float(np.var(yg[region_g == half]))
        var_f = float(np.var(yf[region_f == half]))
        assert abs(var_g - var_f) < TOL, (half, var_g, var_f)


def test_toy_g_flat_is_uniform_across_finer_bins():
    # negative control: residual-from-linear variance must be flat across x, not just matched in
    # the two coarse halves — bin finer (sextiles) and check no drift (the bypass-confound lesson).
    a, sigma = 1.5, 0.25
    x, y = cl.make_toy_g_flat(n=N_LARGE, a=a, sigma=sigma, seed=2)
    xr = x.ravel()
    residual = y - a * xr
    edges = np.quantile(xr, np.linspace(0.0, 1.0, 7))
    bin_vars = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (xr >= lo) & (xr <= hi)
        bin_vars.append(float(np.var(residual[mask])))
    bin_vars = np.array(bin_vars)
    assert np.all(np.abs(bin_vars - sigma**2) < TOL), bin_vars


def test_toy_h_sigma_matches_spec():
    x = np.array([-1.0, -0.001, 0.0, 0.001, 1.0])
    sig = cl.toy_h_sigma(x, sigma_low=0.1, sigma_high=0.8)
    assert list(sig) == [0.1, 0.1, 0.8, 0.8, 0.8]


def test_toy_h_residual_variance_matches_sigma_by_half():
    sigma_low, sigma_high = 0.1, 0.8
    x, y = cl.make_toy_h(n=N_LARGE, sigma_low=sigma_low, sigma_high=sigma_high, seed=3)
    xr = x.ravel()
    f = cl.toy_h_f(xr)
    residual = y - f
    region = cl.toy_g_region(xr)
    var_lo = float(np.var(residual[region == 0]))
    var_hi = float(np.var(residual[region == 1]))
    assert abs(var_lo - sigma_low**2) < TOL, var_lo
    assert abs(var_hi - sigma_high**2) < TOL, var_hi


def test_toy_h_mean_function_identical_across_halves():
    # the FIXED function itself must not depend on region — only sigma does
    x = np.linspace(-1.0, 1.0, 401)
    f = cl.toy_h_f(x, omega=2 * np.pi, amplitude=1.0)
    manual = np.sin(2 * np.pi * x)
    assert np.allclose(f, manual, atol=1e-10)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
