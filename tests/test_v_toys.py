"""Ground-truth tests for the WS3 variance toys (V-toy0, V-toy1).

Direct-runnable (repo-wide pytest collection is broken by an omegaconf conflict):
    ~/dev/.venv/bin/python tests/test_v_toys.py

These verify the toys are the instruments V0/V1/V2 assume: V-toy1 actually plants
f(x)=sin(2πx) and the heteroscedastic σ(x)=0.1+0.3·sigmoid(4x), and V-toy0 is an
exactly well-specified linear-Gaussian model (so the (N−p)/N MLE-variance bias and
its RSS/(N−p) correction are analytic).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "automl_package", "examples"))

import _toy_datasets as td


def test_shapes() -> None:
    x1, y1 = td.make_v_toy1(n=256, seed=0)
    assert x1.shape == (256, 1) and y1.shape == (256,), f"v_toy1 shapes {x1.shape}, {y1.shape}"
    x0, y0 = td.make_v_toy0(n=256, seed=0)
    assert x0.shape == (256, td.V_TOY0_W.size) and y0.shape == (256,), f"v_toy0 shapes {x0.shape}, {y0.shape}"


def test_v_toy1_f_matches_spec() -> None:
    x = np.linspace(0.0, 1.0, 101)
    assert np.allclose(td.v_toy1_f(x), np.sin(2 * np.pi * x), atol=1e-9), "v_toy1_f must equal sin(2π x)"


def test_v_toy1_sigma_matches_spec() -> None:
    x = np.linspace(0.0, 1.0, 101)
    expected = 0.1 + 0.3 / (1.0 + np.exp(-4.0 * x))
    assert np.allclose(td.v_toy1_sigma(x), expected, atol=1e-9), "v_toy1_sigma must equal 0.1 + 0.3·sigmoid(4x)"
    # heteroscedastic: σ genuinely rises with x (not constant)
    assert td.v_toy1_sigma(np.array([1.0]))[0] > td.v_toy1_sigma(np.array([0.0]))[0] + 0.1


def test_v_toy1_heteroscedasticity_recovered() -> None:
    """At large N, the empirical residual std within x-bins tracks the planted σ(x)."""
    x, y = td.make_v_toy1(n=200_000, seed=1)
    xr = x.ravel()
    resid = y - td.v_toy1_f(xr)
    edges = np.linspace(0.0, 1.0, 6)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (xr >= lo) & (xr < hi)
        emp = float(resid[m].std())
        tru = float(td.v_toy1_sigma(np.array([(lo + hi) / 2.0]))[0])
        assert abs(emp - tru) < 0.03, f"bin [{lo:.2f},{hi:.2f}): empirical σ {emp:.3f} vs true {tru:.3f}"


def test_v_toy0_well_specified_ols() -> None:
    """OLS recovers V_TOY0_W and the residual variance ≈ σ² at large N (exactly well-specified)."""
    sigma = 1.3
    x, y = td.make_v_toy0(n=100_000, sigma=sigma, seed=2)
    w_hat, _res, _rank, _sv = np.linalg.lstsq(x, y, rcond=None)
    assert np.allclose(w_hat, td.V_TOY0_W, atol=0.05), f"OLS w_hat {w_hat} vs true {td.V_TOY0_W}"
    resid = y - x @ w_hat
    unbiased = float(resid @ resid) / (x.shape[0] - x.shape[1])  # RSS/(N-p)
    assert abs(unbiased - sigma**2) < 0.05, f"unbiased σ² {unbiased:.4f} vs true {sigma**2:.4f}"


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
            n += 1
    print(f"all {n} tests passed")
