"""Fisher's Z test for conditional independence under linear-Gaussian assumptions.

Fast and well-behaved when relationships are linear and noise is Gaussian.
For nonlinear or non-Gaussian data, prefer :class:`KCITest`.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from cdans.ci_tests.base import _as_2d


class FisherZ:
    """Fisher's Z conditional independence test (partial correlation based)."""

    name = "fisherz"

    def pvalue(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None = None,
    ) -> float:
        x = _as_2d(x)
        y = _as_2d(y)
        if x.shape[1] != 1 or y.shape[1] != 1:
            raise ValueError(
                f"FisherZ requires 1D x and y, got shapes {x.shape}, {y.shape}"
            )
        n = x.shape[0]

        if z is None or (hasattr(z, "size") and z.size == 0):
            r = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
            df = n - 2
        else:
            z = _as_2d(z)
            if z.shape[0] != n:
                raise ValueError(
                    f"sample size mismatch: x has {n}, z has {z.shape[0]}"
                )
            data = np.column_stack([x, y, z])
            try:
                cov = np.cov(data, rowvar=False)
                precision = np.linalg.inv(cov + 1e-10 * np.eye(cov.shape[0]))
            except np.linalg.LinAlgError:
                return 1.0
            # partial correlation between x (col 0) and y (col 1) given z
            diag_prod = float(precision[0, 0] * precision[1, 1])
            if diag_prod <= 0:
                # Numerically degenerate (collinear or extremely ill-conditioned);
                # treat as no information => fail to reject independence.
                return 1.0
            denom = np.sqrt(diag_prod)
            r = float(-precision[0, 1] / denom)
            df = n - z.shape[1] - 2

        if df <= 0:
            return 1.0
        # Numerical guard: clip to avoid Inf in arctanh
        r = float(np.clip(r, -0.9999999, 0.9999999))
        z_stat = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(df - 1)
        # two-sided p-value under standard normal
        return float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
