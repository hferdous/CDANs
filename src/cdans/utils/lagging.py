"""Helpers for building lagged design matrices from time series."""

from __future__ import annotations

import numpy as np


def lagged_design_matrix(
    data: np.ndarray, tau_max: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Build a lagged design matrix from a time series.

    For ``data`` of shape ``(T, n)`` and ``tau_max = k``, returns:

    * ``Y`` of shape ``(T - k, n)``: the "current" values ``X[t]`` for
      ``t = k, k+1, ..., T-1``.
    * ``X_lagged`` of shape ``(T - k, n * k)``: the lagged values, columns
      ordered as ``X_0[t-1], X_1[t-1], ..., X_{n-1}[t-1], X_0[t-2], ...``.
    * ``column_index``: list of ``(variable, lag)`` tuples describing each
      column of ``X_lagged``.

    Parameters
    ----------
    data:
        Time series, shape ``(T, n)`` with ``T > tau_max``.
    tau_max:
        Maximum lag to include.

    Returns
    -------
    tuple
        ``(Y, X_lagged, column_index)``
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    T, n = data.shape
    if tau_max < 1:
        raise ValueError(f"tau_max must be >= 1, got {tau_max}")
    if T <= tau_max:
        raise ValueError(
            f"time series too short: T={T} but need T > tau_max={tau_max}"
        )

    Y = data[tau_max:]
    n_rows = T - tau_max

    column_index: list[tuple[int, int]] = []
    cols: list[np.ndarray] = []
    for lag in range(1, tau_max + 1):
        for var in range(n):
            cols.append(data[tau_max - lag : T - lag, var])
            column_index.append((var, lag))
    X_lagged = np.column_stack(cols)
    assert X_lagged.shape == (n_rows, n * tau_max)
    return Y, X_lagged, column_index


def column_for(var: int, lag: int, n_vars: int) -> int:
    """Return the column index in a lagged design matrix for ``(var, lag)``."""
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    return (lag - 1) * n_vars + var
