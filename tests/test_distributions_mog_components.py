"""Regression tests for MixtureOfGaussiansDistribution public-component access.

Bug: `probreg_identifiability_sweep._compute_metrics` reads `dist.means`,
`dist.stds`, `dist.weights`. These didn't exist as public properties, only as
`_means`/`_stds`/`_weights`. AttributeError was silenced by a bare
`except:` in the sweep, producing NaN for every cell's `nll_mdn_mean` in
the summary CSV — breaking the MDN-likelihood evaluation.
"""

import numpy as np

from automl_package.utils.distributions import MixtureOfGaussiansDistribution


def test_mog_exposes_weights_means_stds_as_public_properties():
    dist = MixtureOfGaussiansDistribution(
        weights=np.array([[0.3, 0.7], [0.5, 0.5]]),
        means=np.array([[1.0, 2.0], [3.0, 4.0]]),
        stds=np.array([[0.1, 0.2], [0.3, 0.4]]),
    )
    # Public read access — required by identifiability sweep.
    assert hasattr(dist, "weights"), "dist.weights missing"
    assert hasattr(dist, "means"), "dist.means missing"
    assert hasattr(dist, "stds"), "dist.stds missing"

    np.testing.assert_allclose(dist.weights, [[0.3, 0.7], [0.5, 0.5]])
    np.testing.assert_allclose(dist.means, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(dist.stds, [[0.1, 0.2], [0.3, 0.4]])


def test_mog_single_sample_components_reshaped():
    """Verify single-sample constructor also exposes public properties with shape (1, K)."""
    dist = MixtureOfGaussiansDistribution(
        weights=np.array([0.4, 0.6]),
        means=np.array([1.0, 2.0]),
        stds=np.array([0.1, 0.2]),
    )
    assert dist.weights.shape == (1, 2)
    assert dist.means.shape == (1, 2)
    assert dist.stds.shape == (1, 2)
