"""Tests for cdans.utils.lagging."""

import numpy as np
import pytest

from cdans.utils.lagging import column_for, lagged_design_matrix


def test_lagged_design_matrix_shapes():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((20, 3))
    Y, X_lagged, idx = lagged_design_matrix(data, tau_max=2)
    assert Y.shape == (18, 3)
    assert X_lagged.shape == (18, 6)
    assert len(idx) == 6


def test_lagged_design_matrix_alignment():
    # Build deterministic data so we can check column contents directly.
    data = np.arange(20).reshape(-1, 1).astype(float)  # column = [0, 1, ..., 19]
    Y, X_lagged, idx = lagged_design_matrix(data, tau_max=2)
    # Y[t] should be data[t + 2]
    np.testing.assert_array_equal(Y[:, 0], np.arange(2, 20))
    # X_lagged column for (var=0, lag=1) is data[t + 1] = Y[t] - 1
    col_lag1 = idx.index((0, 1))
    np.testing.assert_array_equal(X_lagged[:, col_lag1], np.arange(1, 19))
    col_lag2 = idx.index((0, 2))
    np.testing.assert_array_equal(X_lagged[:, col_lag2], np.arange(0, 18))


def test_column_for_inverse():
    n = 4
    for var in range(n):
        for lag in range(1, 4):
            col = column_for(var, lag, n)
            # Recompute via lagged_design_matrix
            data = np.arange(50 * n).reshape(50, n).astype(float)
            _, X_lagged, idx = lagged_design_matrix(data, tau_max=3)
            assert idx[col] == (var, lag)


def test_lagged_design_matrix_too_short():
    with pytest.raises(ValueError, match="too short"):
        lagged_design_matrix(np.zeros((2, 3)), tau_max=3)


def test_lagged_design_matrix_bad_shape():
    with pytest.raises(ValueError):
        lagged_design_matrix(np.zeros(10), tau_max=1)
