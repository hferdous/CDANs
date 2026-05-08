"""Step 3 of CDANs: prune contemporaneous edges and detect changing modules.

This corresponds to Section 3.3 of Ferdous et al. (2023) and contains the
**central methodological contribution** of the paper. The key insight is:

    To test whether a contemporaneous edge ``X_i -- X_j`` should be kept,
    we condition on the **union of the lagged parents of X_i and X_j**
    (and a small number of contemporaneous neighbors), rather than on
    every other contemporaneous variable as standard PC would do.

This bounds the conditioning-set size by ``|Pa_lagged(X_i) ∪ Pa_lagged(X_j)|``
plus a small constant, which is typically much smaller than ``n`` and
therefore both faster and more statistically robust.

The same trick applies to detecting changing modules: ``C -> X_i`` is kept
iff ``X_i`` is dependent on ``C`` even after conditioning on its lagged
parents and contemporaneous neighbors.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from cdans.ci_tests import CITest, get_ci_test
from cdans.graph.timeseries_graph import TimeSeriesGraph
from cdans.utils.lagging import column_for, lagged_design_matrix


def refine_skeleton(
    graph: TimeSeriesGraph,
    data: np.ndarray,
    *,
    surrogate: np.ndarray | None = None,
    ci_test: str | CITest = "fisherz",
    alpha: float = 0.05,
    max_extra_conds: int = 2,
    verbose: bool = False,
) -> TimeSeriesGraph:
    """Prune contemporaneous edges and confirm changing modules.

    Two passes:

    1. **Contemporaneous skeleton refinement.** For each undirected pair
       ``(i, j)`` already in the graph, run CI tests with conditioning sets
       drawn from the lagged parents of ``i`` and ``j``, plus up to
       ``max_extra_conds`` of their contemporaneous neighbors. Drop the
       edge if any test gives a p-value above ``alpha``. Witness sets are
       stored on the graph for use by Step 4 (v-structure detection).

    2. **Changing-modules confirmation.** For each variable ``i`` initially
       marked as changing in Step 2, test ``X_i ⊥ C | lagged_parents(i)``.
       If the test passes (independence not rejected), ``i`` is unmarked.

    Parameters
    ----------
    graph:
        Graph from Step 2 (lagged edges + fully-connected contemporaneous
        skeleton + all variables marked changing).
    data:
        Time series, shape ``(n_samples, n_vars)``.
    surrogate:
        Surrogate variable ``C`` of shape ``(n_samples,)`` or
        ``(n_samples, 1)``. If ``None``, the time index ``[0, 1, ..., T-1]``
        is used (appropriate for nonstationary single-domain data).
    ci_test:
        CI test name or instance.
    alpha:
        Significance level.
    max_extra_conds:
        Maximum number of contemporaneous neighbors to include in the
        conditioning set in addition to the lagged parents. Bigger values
        give stronger tests but at quadratic cost.
    verbose:
        Print per-edge progress.

    Returns
    -------
    TimeSeriesGraph
        The same graph with edges pruned and changing modules confirmed.
        A ``witness`` attribute is attached as ``graph._witness_sets``.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    n_samples, n_vars = data.shape
    if n_vars != graph.n_vars:
        raise ValueError(
            f"data has {n_vars} variables but graph has {graph.n_vars}"
        )

    ci = get_ci_test(ci_test)
    tau_max = graph.tau_max

    Y, X_lagged, _col_index = lagged_design_matrix(data, tau_max)
    n_eff = Y.shape[0]

    if surrogate is None:
        c = np.arange(n_eff, dtype=float).reshape(-1, 1)
    else:
        surrogate = np.asarray(surrogate, dtype=float).reshape(-1, 1)
        if surrogate.shape[0] == n_samples:
            c = surrogate[tau_max:]
        elif surrogate.shape[0] == n_eff:
            c = surrogate
        else:
            raise ValueError(
                f"surrogate has length {surrogate.shape[0]} but expected "
                f"{n_samples} or {n_eff}"
            )

    # --- Pass 1: contemporaneous skeleton refinement ----------------------
    witness_sets: dict[tuple[int, int], list[int]] = {}

    for i, j in list(graph.undirected_contemp_edges()):
        cond_cols = _build_lagged_cond_columns(graph, i, j, n_vars)
        # Plus a few contemporaneous neighbors as additional conditioning
        contemp_pool = sorted(
            (graph.contemp_neighbors(i) | graph.contemp_neighbors(j)) - {i, j}
        )

        edge_kept = True
        # First test: only on lagged parents
        z = X_lagged[:, cond_cols] if cond_cols else None
        p = ci.pvalue(Y[:, i], Y[:, j], z)
        if verbose:
            print(f"  [Step 3] X{i} -- X{j} | lagged({len(cond_cols)} cols): p={p:.4f}")
        if p > alpha:
            graph.remove_contemp_edge(i, j)
            witness_sets[(min(i, j), max(i, j))] = []
            edge_kept = False

        # Then test with subsets of contemporaneous neighbors
        if edge_kept and contemp_pool:
            for k in range(1, min(max_extra_conds, len(contemp_pool)) + 1):
                if not edge_kept:
                    break
                for extra in combinations(contemp_pool, k):
                    extra_data = Y[:, list(extra)]
                    if cond_cols:
                        z = np.column_stack([X_lagged[:, cond_cols], extra_data])
                    else:
                        z = extra_data
                    p = ci.pvalue(Y[:, i], Y[:, j], z)
                    if verbose:
                        print(
                            f"  [Step 3] X{i} -- X{j} | lagged + contemp{list(extra)}: "
                            f"p={p:.4f}"
                        )
                    if p > alpha:
                        graph.remove_contemp_edge(i, j)
                        # Witness records the contemporaneous variables
                        # (lagged conditioning is implicit and always present).
                        witness_sets[(min(i, j), max(i, j))] = list(extra)
                        edge_kept = False
                        break

    # --- Pass 2: confirm changing modules ---------------------------------
    for i in list(graph.changing_modules):
        cond_cols = sorted(
            {column_for(v, lag, n_vars) for (v, lag) in graph.lagged_parents(i)}
        )
        z = X_lagged[:, cond_cols] if cond_cols else None
        p = ci.pvalue(Y[:, i], c, z)
        if verbose:
            print(f"  [Step 3] X{i} ⊥ C | lagged: p={p:.4f}")
        if p > alpha:
            graph.unmark_changing(i)

    # Stash witness sets on the graph for Step 4
    graph._witness_sets = witness_sets  # type: ignore[attr-defined]
    return graph


def _build_lagged_cond_columns(
    graph: TimeSeriesGraph, i: int, j: int, n_vars: int
) -> list[int]:
    """Return sorted column indices in the lagged design matrix for the
    union of lagged parents of ``X_i`` and ``X_j``.
    """
    cols: set[int] = set()
    for var, lag in graph.lagged_parents(i):
        cols.add(column_for(var, lag, n_vars))
    for var, lag in graph.lagged_parents(j):
        cols.add(column_for(var, lag, n_vars))
    return sorted(cols)
