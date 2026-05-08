"""Kernel utilities for KCI.

Implements the Gaussian (RBF) kernel and the two centering operations
used by the KCI test:

* :func:`center_kernel`: standard double-centering ``K -> H K H`` where
  ``H = I - 1/n``. Used by the unconditional test.
* :func:`center_kernel_regression`: regression-based centering
  ``K -> R K R`` where ``R = epsilon * (K_z + epsilon I)^{-1}``. Used by
  the conditional test to "regress out" the conditioning variable.

Both operations are O(n^2) once the kernel matrix is built and avoid an
explicit ``H`` matrix.

Bandwidth heuristics follow the conventions of the reference
implementation by Zhang et al. 2011 (the bundled MATLAB ``KCI-test``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import pinv
from scipy.spatial.distance import pdist, squareform


def gaussian_kernel(X: NDArray[np.float64], width: float) -> NDArray[np.float64]:
    """Compute the Gaussian (RBF) kernel matrix.

    ``K[i, j] = exp(-0.5 * width * ||X[i] - X[j]||^2)``.

    Following the convention of the reference KCI implementation,
    ``width`` is the *precision* parameter (inverse squared bandwidth),
    not the bandwidth itself: ``width = 1 / sigma^2``.

    Parameters
    ----------
    X:
        Data matrix, shape ``(n, d)``.
    width:
        Precision parameter ``> 0``.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if width <= 0:
        raise ValueError(f"width must be > 0, got {width}")
    sq = squareform(pdist(X, "sqeuclidean"))
    return np.exp(-0.5 * width * sq)


def median_width(X: NDArray[np.float64], max_subsample: int = 1000) -> float:
    """Median heuristic for the Gaussian kernel **precision** parameter.

    Returns ``1 / (sqrt(2) * median(pairwise distances))^2``, with a
    subsampling fallback for large ``n``. This matches the reference
    KCI implementation's ``set_width_median``.

    Parameters
    ----------
    X:
        Data matrix, shape ``(n, d)``.
    max_subsample:
        For ``n > max_subsample``, draw a random subsample of this size
        before computing the median. Keeps the cost bounded.
    """
    bw = median_bandwidth(X, max_subsample=max_subsample)
    return float(1.0 / bw**2)


def median_bandwidth(X: NDArray[np.float64], max_subsample: int = 1000) -> float:
    """Median heuristic for the Gaussian kernel **bandwidth** ``sigma``.

    Returns ``sqrt(2) * median(pairwise distances)``. The two-flavour
    distinction matters in CDANs because the KCI test parameterizes
    its kernels by precision (``1/sigma^2``) while the independent-
    change-principle code uses the bandwidth directly.

    Falls back to ``1.0`` for fully-degenerate data (all points
    identical → all pairwise distances zero).
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    n = X.shape[0]
    if n > max_subsample:
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)[:max_subsample]
        X = X[idx]
    dists = pdist(X, "euclidean")
    nonzero = dists[dists > 0]
    if nonzero.size == 0:
        return 1.0
    median_dist = float(np.median(nonzero))
    if median_dist == 0:
        return 1.0
    return float(np.sqrt(2.0) * median_dist)


def empirical_width_hsic(X: NDArray[np.float64]) -> float:
    """Empirical kernel-width heuristic used by the unconditional KCI test.

    Reproduces the schedule of the reference implementation's
    ``set_width_empirical_hsic`` (sample-size-dependent constants then
    scaled by data dimension).
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    n, d = X.shape
    if n < 200:
        bandwidth = 0.8
    elif n < 1200:
        bandwidth = 0.5
    else:
        bandwidth = 0.3
    theta = 1.0 / bandwidth**2
    return float(theta * d)


def empirical_width_kci(X: NDArray[np.float64]) -> float:
    """Empirical kernel-width heuristic used by the conditional KCI test.

    Reproduces the schedule of the reference implementation's
    ``set_width_empirical_kci`` (sample-size-dependent constants then
    scaled inversely by data dimension).
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    n, d = X.shape
    if n < 200:
        bandwidth = 1.2
    elif n < 1200:
        bandwidth = 0.7
    else:
        bandwidth = 0.4
    theta = 1.0 / bandwidth**2
    return float(theta / max(d, 1))


def center_kernel(K: NDArray[np.float64]) -> NDArray[np.float64]:
    """Double-center a kernel matrix: returns ``H K H`` with ``H = I - 1/n``.

    Computed in O(n^2) without forming ``H`` explicitly. Assumes ``K`` is
    symmetric (so column sums equal row sums).
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be a square matrix, got shape {K.shape}")
    n = K.shape[0]
    col_sums = K.sum(axis=0)
    total = col_sums.sum()
    return K - (col_sums[None, :] + col_sums[:, None]) / n + (total / n**2)


def center_kernel_regression(
    K: NDArray[np.float64],
    K_z: NDArray[np.float64],
    epsilon: float = 1e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Regression-based centering: returns ``(R K R, R)`` where
    ``R = epsilon * (K_z + epsilon * I)^{-1}``.

    This residualizes ``K`` against the variable whose kernel is ``K_z``.
    The factor ``epsilon`` is a regularization parameter; ``1e-3`` matches
    the reference KCI implementation.

    Parameters
    ----------
    K:
        Kernel matrix to be residualized, shape ``(n, n)``.
    K_z:
        Kernel matrix of the conditioning variable, shape ``(n, n)``.
    epsilon:
        Regularization parameter, default ``1e-3``.
    """
    if K.shape != K_z.shape:
        raise ValueError(f"shape mismatch: K {K.shape} vs K_z {K_z.shape}")
    n = K.shape[0]
    R = epsilon * pinv(K_z + epsilon * np.eye(n))
    return R @ K @ R, R
