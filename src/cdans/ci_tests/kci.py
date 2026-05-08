"""Self-contained Kernel-based Conditional Independence (KCI) test.

Implements both the unconditional (HSIC-style) and the conditional
versions of the test introduced in Zhang et al. (2011), with the
gamma approximation to the null distribution. No external causal-
discovery libraries are used: just NumPy + SciPy.

Algorithm
---------

**Unconditional test.** Tests ``X ⊥ Y``.

* Build Gaussian kernel matrices ``K_X`` and ``K_Y`` (with bandwidth
  set by an empirical or median heuristic).
* Center them: ``\\tilde K_X = H K_X H``, similarly for ``Y``,
  with ``H = I - (1/n) 1 1^T``.
* Test statistic: ``V = sum(\\tilde K_X * \\tilde K_Y)`` (the V-statistic
  form of HSIC times ``n``).
* Under ``H_0``, ``V`` is a weighted sum of independent ``chi^2(1)``
  variates. The reference approximates the null with a Gamma whose
  first two moments match: ``mean = trace(K_X) trace(K_Y) / n``,
  ``var = 2 sum(K_X^2) sum(K_Y^2) / n^2``.

**Conditional test.** Tests ``X ⊥ Y | Z``.

* Z-score the inputs (matching the reference normalization).
* Build the X-side kernel from ``[X, 0.5 Z]`` concatenated, and the
  Y-side kernel from ``Y``. The X-side concatenation lets the kernel
  capture dependencies that would otherwise be hidden after
  residualization.
* Build a centered kernel ``\\tilde K_Z`` for the conditioning variable.
* Residualize: ``K_X^R = R K_X R``, ``K_Y^R = R K_Y R``, where
  ``R = epsilon (K_Z + epsilon I)^{-1}`` (with ``epsilon = 1e-3``).
* Test statistic: ``V = sum(K_X^R * K_Y^R)``.
* Approximate the null via the spectral construction of Zhang et al.:
  build ``UU^T`` from the eigendecompositions of ``K_X^R`` and
  ``K_Y^R``, and fit a Gamma with the same first two moments.

Reference
---------
Zhang, K., Peters, J., Janzing, D., Schölkopf, B. (2011).
*A kernel-based conditional independence test and application in
causal discovery.* In Uncertainty in Artificial Intelligence (UAI).
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.linalg import eigh

from cdans.ci_tests._kernels import (
    center_kernel,
    center_kernel_regression,
    empirical_width_hsic,
    empirical_width_kci,
    gaussian_kernel,
    median_width,
)
from cdans.ci_tests.base import _as_2d


class KCITest:
    """Kernel-based Conditional Independence test (Zhang et al., 2011).

    Suitable for both linear and nonlinear, Gaussian and non-Gaussian
    relationships. Substantially slower than :class:`FisherZ` —
    ``O(n^3)`` per call — so prefer ``FisherZ`` for linear-Gaussian
    benchmarks and reach for KCI when the residual structure is
    plausibly nonlinear.

    Parameters
    ----------
    width_heuristic:
        How to set the Gaussian kernel bandwidth. ``"empirical"``
        (default) uses the sample-size-dependent constants from the
        reference KCI implementation. ``"median"`` uses the median
        pairwise-distance heuristic. ``"manual"`` requires
        ``width`` to be passed.
    width:
        Manual precision parameter ``1 / sigma^2``. Used only when
        ``width_heuristic="manual"``.
    epsilon:
        Regularization for the conditional residualization step.
        ``1e-3`` matches the reference.
    """

    name = "kci"

    def __init__(
        self,
        width_heuristic: str = "empirical",
        width: float | None = None,
        epsilon: float = 1e-3,
    ) -> None:
        if width_heuristic not in {"empirical", "median", "manual"}:
            raise ValueError(
                f"width_heuristic must be 'empirical', 'median', or 'manual'; "
                f"got {width_heuristic!r}"
            )
        if width_heuristic == "manual" and width is None:
            raise ValueError("width must be provided when width_heuristic='manual'")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._heuristic = width_heuristic
        self._manual_width = width
        self._epsilon = float(epsilon)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pvalue(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64] | None = None,
    ) -> float:
        x = _as_2d(np.asarray(x, dtype=float))
        y = _as_2d(np.asarray(y, dtype=float))
        if z is None or (hasattr(z, "size") and z.size == 0):
            return self._pvalue_unconditional(x, y)
        z = _as_2d(np.asarray(z, dtype=float))
        if z.shape[0] != x.shape[0]:
            raise ValueError(
                f"sample size mismatch: x has {x.shape[0]}, z has {z.shape[0]}"
            )
        return self._pvalue_conditional(x, y, z)

    # ------------------------------------------------------------------
    # Unconditional
    # ------------------------------------------------------------------

    def _pvalue_unconditional(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> float:
        # Z-score (ddof=1 to match reference).
        x = _zscore(x)
        y = _zscore(y)

        wx = self._resolve_width(x, mode="hsic")
        wy = self._resolve_width(y, mode="hsic")

        Kx = gaussian_kernel(x, wx)
        Ky = gaussian_kernel(y, wy)
        Kxc = center_kernel(Kx)
        Kyc = center_kernel(Ky)

        # V-statistic.
        V = float(np.sum(Kxc * Kyc))

        # Gamma null approximation.
        n = Kx.shape[0]
        mean = float(np.trace(Kxc) * np.trace(Kyc) / n)
        var = float(2.0 * np.sum(Kxc**2) * np.sum(Kyc**2) / (n**2))
        return _gamma_pvalue(V, mean, var)

    # ------------------------------------------------------------------
    # Conditional
    # ------------------------------------------------------------------

    def _pvalue_conditional(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
    ) -> float:
        x = _zscore(x)
        y = _zscore(y)
        z = _zscore(z)

        # X-side kernel built from [X, 0.5 Z] concatenation (per the
        # reference, this allows residualization to be effective for
        # rich Z dependencies).
        x_aug = np.concatenate([x, 0.5 * z], axis=1)

        # Bandwidths.
        if self._heuristic == "median":
            wx = median_width(x_aug)
            wy = median_width(y)
            wz = median_width(z)
        elif self._heuristic == "empirical":
            wx = empirical_width_kci(z)
            wy = empirical_width_kci(z)
            wz = empirical_width_kci(z)
        else:  # manual
            assert self._manual_width is not None
            wx = wy = wz = float(self._manual_width)

        Kx = gaussian_kernel(x_aug, wx)
        Ky = gaussian_kernel(y, wy)
        Kz = gaussian_kernel(z, wz)
        # The reference centers Kx and Ky before residualization (KCI.py
        # lines 394–396 of causal-learn). Skipping this step makes the
        # V-statistic explode for conditionally-independent data.
        Kx = center_kernel(Kx)
        Ky = center_kernel(Ky)
        Kz_centered = center_kernel(Kz)

        KxR, _Rz = center_kernel_regression(Kx, Kz_centered, self._epsilon)
        KyR, _ = center_kernel_regression(Ky, Kz_centered, self._epsilon)

        V = float(np.sum(KxR * KyR))

        mean, var = _conditional_null_moments(KxR, KyR)
        return _gamma_pvalue(V, mean, var)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_width(self, X: NDArray[np.float64], *, mode: str) -> float:
        if self._heuristic == "manual":
            assert self._manual_width is not None
            return float(self._manual_width)
        if self._heuristic == "median":
            return median_width(X)
        # empirical
        if mode == "hsic":
            return empirical_width_hsic(X)
        return empirical_width_kci(X)


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _zscore(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-column z-score with ``ddof=1``, treating constant columns as zeros.

    Matches the reference KCI's normalization step.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Z = stats.zscore(X, ddof=1, axis=0)
    Z = np.asarray(Z, dtype=float)
    Z[np.isnan(Z)] = 0.0
    return Z


def _gamma_pvalue(stat: float, mean: float, var: float) -> float:
    """One-sided p-value from a Gamma(k, theta) approximation matching
    the first two moments of the null distribution.

    Returns ``1.0`` (fail to reject) if the moments are degenerate.
    """
    if not np.isfinite(stat) or not np.isfinite(mean) or not np.isfinite(var):
        return 1.0
    if mean <= 0 or var <= 0:
        return 1.0
    k = mean**2 / var
    theta = var / mean
    return float(1.0 - stats.gamma.cdf(stat, a=k, scale=theta))


def _conditional_null_moments(
    KxR: NDArray[np.float64], KyR: NDArray[np.float64], thresh: float = 1e-5
) -> tuple[float, float]:
    """Compute (mean, var) of the gamma approximation for the conditional KCI null.

    Builds the eigenvector-product matrix ``UU`` from the spectral
    decompositions of ``KxR`` and ``KyR``, takes its outer product,
    and reports the trace and trace-of-square — i.e. the first two
    moments of the weighted ``chi^2`` null.

    Implementation follows the reference (Zhang et al., 2011), with
    eigenvalue thresholding to drop near-zero modes.
    """
    n = KxR.shape[0]
    wx, vx = eigh(0.5 * (KxR + KxR.T))
    wy, vy = eigh(0.5 * (KyR + KyR.T))

    # Sort descending.
    idx_x = np.argsort(-wx)
    wx, vx = wx[idx_x], vx[:, idx_x]
    idx_y = np.argsort(-wy)
    wy, vy = wy[idx_y], vy[:, idx_y]

    # Drop tiny eigenvalues (numerical noise).
    if wx.size:
        keep_x = wx > max(wx[0] * thresh, 0.0)
        wx, vx = wx[keep_x], vx[:, keep_x]
    if wy.size:
        keep_y = wy > max(wy[0] * thresh, 0.0)
        wy, vy = wy[keep_y], vy[:, keep_y]

    if wx.size == 0 or wy.size == 0:
        return 0.0, 0.0

    # Scale eigenvectors by sqrt(eigenvalue) — a "feature map" form.
    vx = vx * np.sqrt(wx)[None, :]
    vy = vy * np.sqrt(wy)[None, :]

    num_eigx = vx.shape[1]
    num_eigy = vy.shape[1]
    size_u = num_eigx * num_eigy
    UU = np.zeros((n, size_u))
    for i in range(num_eigx):
        for j in range(num_eigy):
            UU[:, i * num_eigy + j] = vx[:, i] * vy[:, j]

    # Use whichever product is smaller.
    if size_u > n:
        uu_prod = UU @ UU.T
    else:
        uu_prod = UU.T @ UU

    mean = float(np.trace(uu_prod))
    var = float(2.0 * np.trace(uu_prod @ uu_prod))
    return mean, var
