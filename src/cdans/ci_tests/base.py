"""Conditional independence test interface.

CDANs is a constraint-based algorithm: every adjacency decision reduces to
"is X independent of Y given Z?". The interface below abstracts that single
operation so different statistical tests (Fisher's Z, KCI, partial
correlation, …) can be plugged in without touching the search code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class CITest(Protocol):
    """Protocol that every conditional-independence test must implement."""

    name: str

    def pvalue(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None = None,
    ) -> float:
        """Return the p-value of testing ``X ⊥ Y | Z``.

        Parameters
        ----------
        x:
            ``(n_samples,)`` or ``(n_samples, d_x)`` array.
        y:
            ``(n_samples,)`` or ``(n_samples, d_y)`` array.
        z:
            ``(n_samples, d_z)`` array of conditioning variables, or ``None``
            for an unconditional test.

        Returns
        -------
        float
            p-value in ``[0, 1]``. Larger values mean stronger evidence for
            independence.
        """
        ...


def _as_2d(a: np.ndarray) -> np.ndarray:
    """Reshape a 1D array to ``(n, 1)`` so all tests can assume 2D inputs."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    raise ValueError(f"expected 1D or 2D array, got shape {a.shape}")
