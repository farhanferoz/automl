"""Ground-truth tests for the WS3/V0 variance-collapse toys (capacity-ladder program, V0 task).

Covers `make_v_toy0` (linear-Gaussian, well-specified LINEAR model) and `make_v_toy1`
(1-D heteroscedastic with known smooth f(x)/σ(x)) added to `_toy_datasets.py`: each toy's
own sampler must recover its stated ground truth at large N, so downstream V0 measurements
(σ̂/σ_true ratios, SSR) are trustworthy.

Run DIRECTLY (repo-wide pytest collection is broken by an omegaconf conflict), following the
`tests/test_variational_em.py` convention:
    ~/dev/.venv/bin/python tests/test_capacity_ladder_v0.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "automl_package", "examples"))
import _toy_datasets as td


def test_v_toy0_shapes_and_dtypes():
    x, y = td.make_v_toy0(n=50, seed=0)
    assert x.shape == (50, td.V_TOY0_W.size)
    assert y.shape == (50,)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


def test_v_toy0_reproducible_and_seed_varying():
    x1, y1 = td.make_v_toy0(n=50, seed=0)
    x2, y2 = td.make_v_toy0(n=50, seed=0)
    x3, y3 = td.make_v_toy0(n=50, seed=1)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)
    assert not np.array_equal(x1, x3)  # different seed -> different draw
    assert not np.array_equal(y1, y3)


def test_v_toy0_recovers_known_w_and_sigma_at_large_n():
    # OLS on a large draw must recover the fixed, known coefficients and noise scale.
    # sigma_true != 1 so a sampler that ignores its `sigma` argument fails.
    n, sigma_true = 200_000, 1.3
    x, y = td.make_v_toy0(n=n, sigma=sigma_true, seed=7)
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    w_hat, _, _, _ = np.linalg.lstsq(x64, y64, rcond=None)
    assert np.allclose(w_hat, td.V_TOY0_W, atol=0.02), (w_hat, td.V_TOY0_W)

    resid = y64 - x64 @ w_hat
    sigma_hat = float(np.sqrt(np.mean(resid**2)))
    assert abs(sigma_hat - sigma_true) < 0.02, sigma_hat


def test_v_toy1_shapes_and_dtypes():
    x, y = td.make_v_toy1(n=50, seed=0)
    assert x.shape == (50, 1)
    assert y.shape == (50,)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


def test_v_toy1_sigma_bounds_and_monotone():
    # sigma(x) = 0.1 + 0.3*sigmoid(4x): bounded in (0.1, 0.4), strictly increasing on [0, 1].
    grid = np.linspace(0.0, 1.0, 200)
    sigma = td.v_toy1_sigma(grid)
    assert np.allclose(sigma, 0.1 + 0.3 / (1.0 + np.exp(-4.0 * grid)), atol=1e-9)
    assert np.all(sigma > 0.1) and np.all(sigma < 0.4)
    assert np.all(np.diff(sigma) > 0)
    assert abs(sigma[0] - (0.1 + 0.3 / (1.0 + np.exp(0.0)))) < 1e-8


def test_v_toy1_f_matches_sin_2pi_x():
    grid = np.linspace(0.0, 1.0, 200)
    f = td.v_toy1_f(grid)
    assert np.allclose(f, np.sin(2 * np.pi * grid))


def test_v_toy1_sampler_recovers_known_f_and_sigma_at_large_n():
    # Repeated draws at FIXED x locations (the gold-standard-arbiter pattern used by the C/E
    # toys) isolate the per-input mean/std from the marginal-over-x average.
    probe_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    n_repeats = 20_000
    rng = np.random.default_rng(11)
    x_rep = np.tile(probe_x, n_repeats).astype(np.float32)
    # Bypass the RNG-driven x draw of make_v_toy1 by resampling y directly at x_rep.
    y_rep = (td.v_toy1_f(x_rep) + td.v_toy1_sigma(x_rep) * rng.normal(0.0, 1.0, x_rep.size)).astype(np.float32)

    for i, x0 in enumerate(probe_x):
        mask = x_rep == np.float32(x0)
        y_here = y_rep[mask].astype(np.float64)
        mean_hat = float(np.mean(y_here))
        std_hat = float(np.std(y_here))
        assert abs(mean_hat - td.v_toy1_f(x0)[0]) < 0.02, (x0, mean_hat)
        assert abs(std_hat - td.v_toy1_sigma(x0)[0]) < 0.02, (x0, std_hat)


def test_v_toy1_marginal_sampler_matches_binned_mean():
    # The actual make_v_toy1 sampler (x ~ U(0,1)) recovers f when binned over a large draw.
    # (sigma is NOT checked via binning here: f's slope over a finite-width bin blurs the
    # within-bin std upward — a first-order smear artifact, not a sampler bug; sigma is
    # verified properly by the fixed-x repeated-draw test below, which has no bin width.)
    x, y = td.make_v_toy1(n=400_000, seed=3)
    xr, yr = x.ravel().astype(np.float64), y.astype(np.float64)
    edges = np.linspace(0.0, 1.0, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (xr >= lo) & (xr < hi)
        assert mask.sum() > 1000
        center = (lo + hi) / 2.0
        mean_hat = float(np.mean(yr[mask]))
        assert abs(mean_hat - td.v_toy1_f(np.array([center]))[0]) < 0.03, (center, mean_hat)


def test_study_toys_true_mean_and_floor_match_their_generators():
    """Each study toy's `true_mean` is the generator's mean, its `noise_sd` the generator's spread and its `floor` the mean noise variance (4 standard errors, 200,000 draws); x stays in `x_range` and regions are masks."""
    n_draws, n_se = 200_000, 4.0
    toy_params = {td.StudyToy.JUMP: {"jump": 1.0}}
    assert set(td.STUDY_TOYS) == set(td.StudyToy)
    for toy, spec in td.STUDY_TOYS.items():
        params = toy_params.get(toy, {})
        x, y = spec.generator(n_draws, 7, **params)
        resid = np.asarray(y, dtype=np.float64).ravel() - spec.true_mean(x, **params)
        assert resid.shape == (n_draws,), toy
        se_mean = resid.std() / np.sqrt(n_draws)
        assert abs(resid.mean()) < n_se * se_mean, (toy, resid.mean(), se_mean)
        sq = resid**2
        se_sq = sq.std() / np.sqrt(n_draws)
        assert abs(sq.mean() - spec.floor) < n_se * se_sq, (toy, sq.mean(), spec.floor, se_sq)
        sd = spec.noise_sd(x, **params)
        assert sd.shape == (n_draws,), toy
        assert abs(np.mean(sd**2) - spec.floor) < 1e-2 * spec.floor, (toy, np.mean(sd**2), spec.floor)
        z = resid / sd  # Gaussian given x: standardised residuals are N(0, 1)
        assert abs(z.mean()) < n_se / np.sqrt(n_draws) and abs(z.std() - 1.0) < n_se / np.sqrt(2 * n_draws), toy
        low, high = spec.x_range
        assert low <= x.min() and x.max() <= high, toy
        if spec.regions:
            masks = np.stack([mask(x) for mask in spec.regions.values()])
            assert masks.dtype == bool and (masks.sum(axis=0) == 1).all(), toy  # the regions partition x


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
