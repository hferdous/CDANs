"""Conditional independence tests for CDANs."""

from __future__ import annotations

from cdans.ci_tests.base import CITest
from cdans.ci_tests.fisher_z import FisherZ


def get_ci_test(name: str | CITest, **kwargs) -> CITest:
    """Resolve a CI test by name, or return one already given.

    Parameters
    ----------
    name:
        Either a string (``"fisherz"`` or ``"kci"``) or an object that already
        implements the :class:`CITest` protocol.
    **kwargs:
        Forwarded to the test constructor when ``name`` is a string.

    Returns
    -------
    CITest
        A ready-to-use CI test instance.
    """
    if isinstance(name, str):
        key = name.lower().replace("-", "").replace("_", "")
        if key in {"fisherz", "fisher"}:
            return FisherZ()
        if key in {"kci", "kernelci"}:
            from cdans.ci_tests.kci import KCITest

            return KCITest(**kwargs)
        raise ValueError(f"unknown CI test name: {name!r}")
    if isinstance(name, CITest):
        return name
    raise TypeError(f"expected str or CITest, got {type(name).__name__}")


__all__ = ["CITest", "FisherZ", "get_ci_test"]
