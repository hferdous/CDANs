"""Tests for cdans.utils.synthetic."""

import numpy as np
import pytest

from cdans.utils.synthetic import generate_synthetic_cdans


def test_basic_generation_shapes():
    ds = generate_synthetic_cdans(n_vars=4, n_samples=200, tau_max=2, seed=0)
    assert ds.data.shape == (200, 4)
    assert ds.n_vars == 4
    assert ds.n_samples == 200
    assert all(lag >= 1 for _, _, lag in ds.lagged_edges)


def test_reproducibility():
    ds1 = generate_synthetic_cdans(n_vars=4, n_samples=150, tau_max=2, seed=42)
    ds2 = generate_synthetic_cdans(n_vars=4, n_samples=150, tau_max=2, seed=42)
    np.testing.assert_array_equal(ds1.data, ds2.data)
    assert ds1.lagged_edges == ds2.lagged_edges
    assert ds1.contemporaneous_edges == ds2.contemporaneous_edges
    assert ds1.changing_modules == ds2.changing_modules


def test_contemp_edges_are_dag():
    ds = generate_synthetic_cdans(n_vars=6, n_samples=200, tau_max=2, seed=1)
    # By construction (i < j ordering) the contemporaneous edges form a DAG.
    for i, j in ds.contemporaneous_edges:
        assert i < j, f"contemp edge ({i}, {j}) violates DAG topological order"


def test_data_is_finite():
    ds = generate_synthetic_cdans(n_vars=8, n_samples=500, tau_max=3, seed=2)
    assert np.all(np.isfinite(ds.data))


def test_validation():
    with pytest.raises(ValueError, match="n_changing"):
        generate_synthetic_cdans(n_vars=2, n_changing=5)
    with pytest.raises(ValueError, match="tau_max"):
        generate_synthetic_cdans(tau_max=0)


def test_changing_modules_count():
    ds = generate_synthetic_cdans(n_vars=6, n_changing=3, n_samples=100, seed=3)
    assert len(ds.changing_modules) == 3
    assert all(0 <= v < 6 for v in ds.changing_modules)


def test_no_self_lag1_overwrite():
    """Auto-regressive (var, var, 1) must always be present."""
    ds = generate_synthetic_cdans(n_vars=5, n_samples=120, tau_max=2, seed=4)
    for v in range(5):
        assert (v, v, 1) in ds.lagged_edges
