"""Meek's rules for orienting edges in a partially-directed acyclic graph.

These four rules iteratively orient edges in a PDAG to enlarge the set of
directed edges while preserving the Markov equivalence class. They were
introduced by Meek (1995) and are a standard component of constraint-based
causal discovery (PC, CD-NOD, CDANs).

This implementation operates on a :class:`TimeSeriesGraph` and orients
**only the contemporaneous edges**. Lagged edges and surrogate edges are
already directed by construction. Only contemporaneous v-structures and
contemporaneous skeleton constraints are used as input; lagged and surrogate
edges are excluded from rule application to keep the rules sound for the
mixed-graph setting (see CDANs §3.4).
"""

from __future__ import annotations

from cdans.graph.timeseries_graph import TimeSeriesGraph


def apply_meek_rules(graph: TimeSeriesGraph, max_iter: int = 100) -> TimeSeriesGraph:
    """Iteratively apply Meek's rules to a contemporaneous PDAG until a fixed point.

    Modifies ``graph`` in place and returns it.

    Rules implemented (over contemporaneous edges only):

    * **R1** — if ``a -> b`` and ``b -- c`` and ``a`` and ``c`` are not adjacent,
      orient ``b -> c`` (avoids new v-structure).
    * **R2** — if ``a -> b -> c`` and ``a -- c``, orient ``a -> c`` (avoids cycle).
    * **R3** — if ``a -- b``, ``a -- c``, ``a -- d``, ``c -> b``, ``d -> b``, and
      ``c`` and ``d`` are not adjacent, orient ``a -> b``.
    * **R4** — if ``a -- b``, ``a -- c``, ``c -> d``, ``d -> b``, and ``a`` and
      ``d`` are not adjacent, orient ``a -> b``.

    Parameters
    ----------
    graph:
        Time-series graph with some contemporaneous edges already oriented.
    max_iter:
        Safety bound on the fixed-point iteration. The algorithm always
        converges in finite steps for finite graphs; this is paranoia.

    Returns
    -------
    TimeSeriesGraph
        The same graph, mutated in place.
    """
    n = graph.n_vars

    for _ in range(max_iter):
        changed = False

        # R1: a -> b, b -- c, a not adj c  =>  b -> c
        for a, b in list(graph.directed_contemp_edges()):
            for c in range(n):
                if c in (a, b):
                    continue
                if not graph.is_contemp_undirected(b, c):
                    continue
                if graph.has_contemp_edge(a, c):
                    continue
                graph.orient_contemp(b, c)
                changed = True

        # R2: a -> b -> c and a -- c  =>  a -> c
        for a, b in list(graph.directed_contemp_edges()):
            for c in range(n):
                if c in (a, b):
                    continue
                if not _is_directed(graph, b, c):
                    continue
                if graph.is_contemp_undirected(a, c):
                    graph.orient_contemp(a, c)
                    changed = True

        # R3: a -- b, c -> b, d -> b, a -- c, a -- d, c not adj d  =>  a -> b
        for a, b in list(graph.undirected_contemp_edges()):
            # try both orientations of the undirected edge as candidate (a -> b)
            for src, dst in [(a, b), (b, a)]:
                parents_of_dst = [
                    v for v in range(n) if v != dst and _is_directed(graph, v, dst)
                ]
                # Need two such parents both connected to src by undirected edges
                # and not adjacent to each other.
                for k1 in range(len(parents_of_dst)):
                    for k2 in range(k1 + 1, len(parents_of_dst)):
                        c, d = parents_of_dst[k1], parents_of_dst[k2]
                        if c == src or d == src:
                            continue
                        if not graph.is_contemp_undirected(src, c):
                            continue
                        if not graph.is_contemp_undirected(src, d):
                            continue
                        if graph.has_contemp_edge(c, d):
                            continue
                        if graph.is_contemp_undirected(src, dst):
                            graph.orient_contemp(src, dst)
                            changed = True
                            break
                    else:
                        continue
                    break

        # R4: a -- b, a -- c, c -> d, d -> b, a not adj d  =>  a -> b
        for a, b in list(graph.undirected_contemp_edges()):
            for src, dst in [(a, b), (b, a)]:
                for c in range(n):
                    if c in (src, dst):
                        continue
                    if not graph.is_contemp_undirected(src, c):
                        continue
                    for d in range(n):
                        if d in (src, dst, c):
                            continue
                        if not _is_directed(graph, c, d):
                            continue
                        if not _is_directed(graph, d, dst):
                            continue
                        if graph.has_contemp_edge(src, d):
                            continue
                        if graph.is_contemp_undirected(src, dst):
                            graph.orient_contemp(src, dst)
                            changed = True

        if not changed:
            break

    return graph


def _is_directed(graph: TimeSeriesGraph, src: int, dst: int) -> bool:
    """True iff there is a directed edge ``src -> dst`` (not undirected)."""
    return bool(graph.contemp_adj[src, dst] == 1 and graph.contemp_adj[dst, src] == 0)
