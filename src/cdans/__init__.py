"""CDANs: Temporal Causal Discovery from Autocorrelated and Non-Stationary Time Series Data.

A Python implementation of the algorithm presented in:

    Ferdous, M. H., Hasan, U., & Gani, M. O. (2023). CDANs: Temporal Causal
    Discovery from Autocorrelated and Non-Stationary Time Series Data.
    Proceedings of the 8th Machine Learning for Healthcare Conference, PMLR 219.

The recommended entry point is the :class:`CDANs` class:

    >>> from cdans import CDANs
    >>> from cdans.utils import generate_synthetic_cdans
    >>> dataset = generate_synthetic_cdans(n_vars=4, n_samples=300, tau_max=2)
    >>> result = CDANs(tau_max=2, alpha=0.05).fit(dataset.data)
    >>> print(result.summary())  # doctest: +SKIP

For finer control, the four steps can be invoked individually from
:mod:`cdans.steps`.
"""

from cdans.algorithm import CDANs, CDANsResult
from cdans.evaluation import (
    GraphMetrics,
    StructureRecoveryMetrics,
    evaluate_graph,
    shd,
)
from cdans.graph import TimeSeriesGraph
from cdans.plotting import plot_process_graph, plot_time_series_graph

__version__ = "0.1.0"

__all__ = [
    "CDANs",
    "CDANsResult",
    "GraphMetrics",
    "StructureRecoveryMetrics",
    "TimeSeriesGraph",
    "__version__",
    "evaluate_graph",
    "plot_process_graph",
    "plot_time_series_graph",
    "shd",
]
