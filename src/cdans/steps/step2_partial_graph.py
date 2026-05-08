"""Step 2 of CDANs: construct the partial undirected graph.

This corresponds to Section 3.2 of Ferdous et al. (2023). Given the lagged
adjacencies discovered in Step 1, we construct a partial graph that:

1. **Keeps** all lagged directed edges from Step 1.
2. **Adds** an undirected contemporaneous edge between every pair of
   variables (initial fully-connected contemporaneous skeleton, to be
   thinned in Step 3).
3. **Marks** the surrogate variable ``C`` as a candidate parent of every
   variable (changing-modules will be filtered in Step 3).

This is a quick step — no statistical tests are run here. It simply prepares
the data structure that Step 3 operates on.
"""

from __future__ import annotations

from cdans.graph.timeseries_graph import TimeSeriesGraph


def build_partial_graph(graph: TimeSeriesGraph) -> TimeSeriesGraph:
    """Add the fully-connected contemporaneous skeleton to ``graph``.

    Modifies ``graph`` in place and returns it. The lagged edges already
    present are preserved untouched.

    Parameters
    ----------
    graph:
        The output of Step 1 (lagged adjacency identification).

    Returns
    -------
    TimeSeriesGraph
        The same graph with all contemporaneous pairs ``(i, j)`` for
        ``i < j`` connected by an undirected edge.
    """
    n = graph.n_vars
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_contemp_undirected(i, j)
    # Tentatively assume every variable could have a changing mechanism;
    # Step 3 will prune via CI tests against the surrogate.
    for i in range(n):
        graph.mark_changing(i)
    return graph
