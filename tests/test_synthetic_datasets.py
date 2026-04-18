"""Sanity checks for synthetic benchmark datasets B1-B6."""

from __future__ import annotations

import numpy as np
import pytest

from automl_package.utils.synthetic_datasets import (
    generate_b1_gravitational,
    generate_b2_oscillator,
    generate_b3_two_phase,
    generate_b4_latent_groups,
    generate_b5_sparse_importance,
    generate_b6_null,
    load_fixture,
)


@pytest.mark.parametrize("gen, expected_n, expected_d", [
    (generate_b1_gravitational, 10_000, 10),
    (generate_b2_oscillator, 10_000, 8),
    (generate_b3_two_phase, 10_000, 12),
    (generate_b4_latent_groups, 10_000, 10),
    (generate_b5_sparse_importance, 10_000, 30),
    (generate_b6_null, 5_000, 8),
])
def test_shape_and_finite(gen, expected_n, expected_d):
    ds = gen()
    assert ds.x.shape == (expected_n, expected_d)
    assert ds.y.shape == (expected_n,)
    assert np.all(np.isfinite(ds.x))
    assert np.all(np.isfinite(ds.y))


def test_b1_has_multimodal_structure():
    """B1: sign of (x_pos - x0) should vary, so the target posterior is bimodal-prone."""
    ds = generate_b1_gravitational(n=500, seed=0)
    # m is log-normal so should cover a range > 1 decade
    assert ds.meta["m"].max() / max(ds.meta["m"].min(), 1e-9) > 3.0


def test_b3_noise_peaks_near_threshold():
    ds = generate_b3_two_phase(n=2_000, seed=0)
    dist = np.abs(ds.meta["projection"])
    sigma = ds.meta["noise_std"]
    # Correlation between distance-from-threshold and noise should be strongly negative.
    r = np.corrcoef(dist, sigma)[0, 1]
    assert r < -0.5, f"noise_std vs distance correlation {r:.2f} not strongly negative"


def test_b4_groups_have_distinct_noise():
    ds = generate_b4_latent_groups(n=5_000, seed=0)
    groups = ds.meta["group"]
    sigma = ds.meta["noise_std"]
    mean_sig_per_group = [sigma[groups == g].mean() for g in (0, 1, 2)]
    assert mean_sig_per_group[0] < mean_sig_per_group[1] < mean_sig_per_group[2]


def test_b5_noise_features_uncorrelated_with_y():
    ds = generate_b5_sparse_importance(n=5_000, seed=0)
    corrs = np.array([np.corrcoef(ds.x[:, i], ds.y)[0, 1] for i in range(ds.x.shape[1])])
    informative = np.abs(corrs[:3])
    noise = np.abs(corrs[3:])
    assert informative.mean() > noise.mean() * 3, "informative features should correlate with y more strongly than noise features"


def test_fixture_roundtrip(tmp_path):
    ds = generate_b6_null(n=200, seed=0)
    ds.to_npz(tmp_path)
    loaded = load_fixture("b6_null", directory=tmp_path)
    np.testing.assert_array_equal(loaded.x, ds.x)
    np.testing.assert_array_equal(loaded.y, ds.y)
