"""Shared test fixtures for AutoML test suite."""

import numpy as np
import pytest


@pytest.fixture
def heteroscedastic_data():
    """Generates heteroscedastic sine data where noise grows with |x|."""
    np.random.seed(42)
    x = np.random.uniform(-5, 5, 500).reshape(-1, 1)
    y_true = np.sin(x) * 2 + 0.5 * x
    noise_std = 0.1 + 0.4 * np.abs(x)
    y = y_true + np.random.normal(0, noise_std)
    return x, y.ravel(), y_true.ravel(), noise_std.ravel()


@pytest.fixture
def multimodal_data():
    """Generates bimodal data: y = x +/- 1.5."""
    np.random.seed(42)
    x = np.random.uniform(-3, 3, 500).reshape(-1, 1)
    sign = np.random.choice([-1, 1], size=500).reshape(-1, 1)
    y = x + sign * 1.5 + np.random.normal(0, 0.1, (500, 1))
    return x, y.ravel()


@pytest.fixture
def simple_linear_data():
    """Generates simple y = 2x + 1 + noise."""
    np.random.seed(42)
    x = np.random.uniform(-5, 5, 300).reshape(-1, 1)
    y = (2 * x + 1 + np.random.normal(0, 0.3, (300, 1))).ravel()
    return x, y


@pytest.fixture
def piecewise_data():
    """Piecewise function: linear for x<0, sinusoidal for x>=0."""
    np.random.seed(42)
    x = np.random.uniform(-5, 5, 500).reshape(-1, 1)
    y_true = np.where(
        x < 0,
        0.5 * x,
        0.5 * x + np.sin(4 * np.pi * x),
    ).ravel()
    y = y_true + np.random.normal(0, 0.2, 500)
    return x, y, y_true
