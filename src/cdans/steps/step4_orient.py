"""Step 4 of CDANs: orient contemporaneous edges.

This corresponds to Section 3.4 of Ferdous et al. (2023). After Step 3 has
produced the contemporaneous skeleton and confirmed which variables have
changing modules, Step 4 determines edge directions in four sub-passes:

1. **Surrogate orientation.** Every changing module ``X_i`` gets a
   directed edge ``C -> X_i``. This is encoded by ``X_i`` being in
   ``changing_modules`` (no contemporaneous edge to ``C`` is
   materialized; the surrogate is treated specially).

2. **V-structure detection.** For every triple ``(a, b, c)`` such that
   ``a -- b -- c`` and ``a`` and ``c`` are not adjacent: orient as
   ``a -> b <- c`` iff ``b`` is **not** in the witness set that
   separated ``a`` and ``c`` in Step 3.

3. **Independent-change-principle sink-finding** (CD-NOD §4 /
   CDANs §3.4). For undirected edges between two changing modules,
   iteratively pick the "most confident sink": the candidate variable
   whose parents and conditional change most independently with the
   surrogate. Orient all that candidate's undirected edges inward,
   remove it from the candidate pool, repeat. See
   :mod:`cdans.independent_change`.

4. **Meek's rules.** Iteratively apply the four standard Meek rules
   to propagate orientations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cdans.graph.meek import apply_meek_rules
from cdans.graph.timeseries_graph import TimeSeriesGraph
from cdans.independent_change import AUTO, WidthSpec, independent_change_score


def orient_edges(
    graph: TimeSeriesGraph,
    *,
    data: NDArray[np.float64] | None = None,
    surrogate: NDArray[np.float64] | None = None,
    use_independent_change: bool = True,
    independent_change_width: WidthSpec = AUTO,
) -> TimeSeriesGraph:
    """Orient contemporaneous edges in the graph in place.

    Parameters
    ----------
    graph:
        Graph from Step 3 (lagged + contemporaneous skeleton + confirmed
        changing modules + ``_witness_sets`` attribute).
    data:
        Time series, shape ``(n_samples, n_vars)``. Required when
        ``use_independent_change=True`` and there are at least two
        connected changing modules with undirected edges.
    surrogate:
        Distribution-shift surrogate, shape ``(n_samples,)`` or
        ``(n_samples, 1)``. Same requirement as ``data``. Truncated
        to match the lagged design matrix length internally.
    use_independent_change:
        Whether to run the independent-change-principle sink-finding
        sub-pass. Setting this to ``False`` returns the Markov
        equivalence class without trying to break ties on undirected
        edges between changing modules.
    independent_change_width:
        Bandwidth for the kernels inside
        :func:`independent_change_score`. ``"auto"`` (default) picks
        per-call via the median heuristic on the standardized parents;
        ``"gp"`` learns the bandwidth via a Gaussian-process fit with an
        ARD-RBF kernel (slower, more adaptive); a positive float forces
        a manual value. The MATLAB reference uses ``0.1``.

    Returns
    -------
    TimeSeriesGraph
        The same graph with as many contemporaneous edges oriented as
        the algorithm can determine.
    """
    witness_sets: dict[tuple[int, int], list[int]] = getattr(
        graph, "_witness_sets", {}
    )

    _orient_v_structures(graph, witness_sets)
    apply_meek_rules(graph)

    if use_independent_change:
        if data is None or surrogate is None:
            # Nothing to do without data; silently skip.
            pass
        else:
            _orient_independent_change(
                graph,
                data=np.asarray(data, dtype=float),
                surrogate=np.asarray(surrogate, dtype=float).reshape(-1),
                width=independent_change_width,
            )
            apply_meek_rules(graph)

    return graph


# ---------------------------------------------------------------------------
# V-structure orientation
# ---------------------------------------------------------------------------


def _orient_v_structures(
    graph: TimeSeriesGraph,
    witness_sets: dict[tuple[int, int], list[int]],
) -> None:
    """Find unshielded colliders and orient them as ``a -> b <- c``."""
    n = graph.n_vars
    for b in range(n):
        nbrs = sorted(graph.contemp_neighbors(b))
        for ai in range(len(nbrs)):
            for ci in range(ai + 1, len(nbrs)):
                a, c = nbrs[ai], nbrs[ci]
                if graph.has_contemp_edge(a, c):
                    continue  # shielded triple, skip

                witness_key = (min(a, c), max(a, c))
                witness = witness_sets.get(witness_key)

                # If b appears in the separating set, it is *not* a collider.
                if witness is not None and b in witness:
                    continue

                # Both edges must currently be undirected to orient.
                if graph.is_contemp_undirected(a, b) and graph.is_contemp_undirected(c, b):
                    graph.orient_contemp(a, b)
                    graph.orient_contemp(c, b)
                # CDANs surrogate-priority tie-breaker on partial cases.
                elif graph.is_contemp_undirected(a, b):
                    if a in graph.changing_modules and b not in graph.changing_modules:
                        graph.orient_contemp(a, b)
                elif graph.is_contemp_undirected(c, b):
                    if c in graph.changing_modules and b not in graph.changing_modules:
                        graph.orient_contemp(c, b)


# ---------------------------------------------------------------------------
# Independent change principle (iterative sink-finding)
# ---------------------------------------------------------------------------


def _orient_independent_change(
    graph: TimeSeriesGraph,
    *,
    data: NDArray[np.float64],
    surrogate: NDArray[np.float64],
    width: float,
) -> None:
    """Iteratively pick the "most confident sink" and orient its parents inward.

    Mirrors the orchestration in ``nonsta_cd_new.m`` (lines 191–219).
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")

    # Align surrogate to the data length used by the score function.
    n_samples = data.shape[0]
    if surrogate.shape[0] != n_samples:
        # Allow the caller to pass the full-length surrogate even if
        # data is shorter; just truncate from the front.
        if surrogate.shape[0] > n_samples:
            surrogate = surrogate[-n_samples:]
        else:
            raise ValueError(
                f"surrogate too short: {surrogate.shape[0]} < {n_samples}"
            )

    # Candidate set: changing modules with at least one undirected edge.
    def has_undirected(v: int) -> bool:
        return any(
            graph.is_contemp_undirected(v, u)
            for u in range(graph.n_vars)
            if u != v
        )

    candidates: list[int] = sorted(v for v in graph.changing_modules if has_undirected(v))

    while len(candidates) > 1:
        scores: list[float] = []
        parent_sets: list[list[int]] = []

        for cand in candidates:
            # Candidate's hypothesized parents = everyone with an edge
            # pointing into `cand` OR an undirected edge to it.
            # In our adjacency convention both have contemp_adj[p, cand] == 1.
            parents = [
                p for p in range(graph.n_vars)
                if p != cand and graph.contemp_adj[p, cand] == 1
            ]
            if not parents:
                # No parents to score against; this candidate cannot
                # be tested this round.
                scores.append(np.inf)
                parent_sets.append([])
                continue
            try:
                score = independent_change_score(
                    parents=data[:, parents],
                    effect=data[:, cand],
                    surrogate=surrogate,
                    width=width,
                )
            except np.linalg.LinAlgError:
                # Numerical failure on this candidate; skip it this round.
                # ValueError (invalid args) is allowed to propagate so the
                # user sees a clear error.
                score = float("inf")
            scores.append(score)
            parent_sets.append(parents)

        if all(np.isinf(s) for s in scores):
            break  # nothing to orient this round

        best_idx = int(np.argmin(scores))
        sink = candidates[best_idx]

        # Orient every candidate parent toward the sink.
        for parent in parent_sets[best_idx]:
            if graph.is_contemp_undirected(parent, sink):
                graph.orient_contemp(parent, sink)

        # Remove the sink from the pool and continue.
        candidates.pop(best_idx)
