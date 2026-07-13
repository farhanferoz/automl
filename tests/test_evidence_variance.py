"""Ground-truth tests for WS3/V1 global-variance-estimation mechanisms.

Checks the four mechanisms ranked in
`docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md` sec 4 (V1) against ANALYTIC truth on
large-N well-specified linear-Gaussian data: (i) in-sample MLE sigma_hat^2 is biased low by the
classical (N-p)/N factor; (ii) the MacKay alpha,beta evidence fixed-point recovers sigma_true^2 and
corrects the MLE bias; (iii)/(iv) held-out and K-fold cross-fitted sigma_hat^2 recover sigma_true^2.
Also checks the new `make_v_toy1h` homoscedastic twin actually holds sigma CONSTANT (unlike
`make_v_toy1`'s planted rising sigma(x)).

Direct-runnable (repo-wide pytest collection is broken by an omegaconf conflict):
    ~/dev/.venv/bin/python tests/test_evidence_variance.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "automl_package", "examples"))

import _toy_datasets as td
from capacity_ladder_variance_v1 import cross_fitted_sigma2, evidence_fixed_point, held_out_sigma2, ols_fit_fn


def test_v_toy1h_shape_and_mean_matches_v_toy1() -> None:
    x, y = td.make_v_toy1h(n=256, seed=0)
    assert x.shape == (256, 1) and y.shape == (256,), f"v_toy1h shapes {x.shape}, {y.shape}"
    xr64 = x.ravel().astype(np.float64)
    assert np.allclose(td.v_toy1_f(x.ravel()), np.sin(2 * np.pi * xr64), atol=1e-6), "v_toy1h must share v_toy1's mean f(x)=sin(2πx)"


def test_v_toy1h_homoscedastic() -> None:
    """Empirical residual std is approx CONSTANT across x-bins (unlike v_toy1's rising sigma(x))."""
    x, y = td.make_v_toy1h(n=200_000, seed=1)
    xr = x.ravel()
    resid = y - td.v_toy1_f(xr)
    edges = np.linspace(0.0, 1.0, 6)
    bin_stds = np.array([float(resid[(xr >= lo) & (xr < hi)].std()) for lo, hi in zip(edges[:-1], edges[1:])])
    assert np.allclose(bin_stds, bin_stds.mean(), atol=0.03), f"bin stds {bin_stds} not homoscedastic"
    assert abs(bin_stds.mean() - td.V_TOY1H_SIGMA) < 0.02, f"mean bin std {bin_stds.mean():.4f} vs planted σ {td.V_TOY1H_SIGMA}"


def test_mle_biased_low() -> None:
    """In-sample MLE sigma_hat^2 = RSS/N is biased low by the classical (N-p)/N factor."""
    sigma, n, p = 1.3, 20_000, td.V_TOY0_W.size
    x, y = td.make_v_toy0(n=n, sigma=sigma, seed=2)
    phi = x.astype(np.float64)
    w_hat, _, _, _ = np.linalg.lstsq(phi, y.astype(np.float64), rcond=None)
    resid = y.astype(np.float64) - phi @ w_hat
    mle_sigma2 = float(np.sum(resid**2) / n)
    expected = (n - p) / n * sigma**2
    assert abs(mle_sigma2 - expected) < 0.05, f"MLE σ² {mle_sigma2:.4f} vs expected biased {expected:.4f}"


def test_evidence_recovers_sigma2_and_beats_mle() -> None:
    """MacKay fixed-point evidence β⁻¹ ≈ σ²_true and corrects the MLE's downward bias."""
    sigma, n = 1.3, 20_000
    x, y = td.make_v_toy0(n=n, sigma=sigma, seed=3)
    phi = x.astype(np.float64)
    yf = y.astype(np.float64)

    ev = evidence_fixed_point(phi, yf)
    assert ev["converged"], f"evidence fixed-point did not converge: {ev}"
    evidence_sigma2 = 1.0 / ev["beta"]
    assert abs(evidence_sigma2 - sigma**2) < 0.05, f"evidence σ² {evidence_sigma2:.4f} vs true {sigma**2:.4f}"

    w_hat, _, _, _ = np.linalg.lstsq(phi, yf, rcond=None)
    resid = yf - phi @ w_hat
    mle_sigma2 = float(np.sum(resid**2) / n)
    assert evidence_sigma2 > mle_sigma2, "evidence σ² must correct the MLE's downward bias"


def test_heldout_and_crossfitted_recover_sigma2() -> None:
    """Held-out-half and 5-fold cross-fitted σ̂² both recover σ²_true on well-specified data."""
    sigma, n = 1.3, 20_000
    x, y = td.make_v_toy0(n=n, sigma=sigma, seed=4)
    fit_fn = ols_fit_fn(lambda xx: xx.astype(np.float64))

    heldout_sigma2 = held_out_sigma2(x, y, fit_fn, seed=0)
    crossfit_sigma2, per_fold_mse = cross_fitted_sigma2(x, y, fit_fn, k=5, seed=0)

    assert abs(heldout_sigma2 - sigma**2) < 0.08, f"held-out σ² {heldout_sigma2:.4f} vs true {sigma**2:.4f}"
    assert abs(crossfit_sigma2 - sigma**2) < 0.05, f"cross-fitted σ² {crossfit_sigma2:.4f} vs true {sigma**2:.4f}"
    assert len(per_fold_mse) == 5, f"expected 5 per-fold MSE values, got {len(per_fold_mse)}"


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
            n += 1
    print(f"all {n} tests passed")
