"""The four algorithmic steps of CDANs.

Each step is a pure function operating on a :class:`~cdans.graph.TimeSeriesGraph`,
making the pipeline easy to compose and inspect.
"""

from cdans.steps.step1_lagged import discover_lagged_adjacencies
from cdans.steps.step2_partial_graph import build_partial_graph
from cdans.steps.step3_skeleton import refine_skeleton
from cdans.steps.step4_orient import orient_edges

__all__ = [
    "build_partial_graph",
    "discover_lagged_adjacencies",
    "orient_edges",
    "refine_skeleton",
]
