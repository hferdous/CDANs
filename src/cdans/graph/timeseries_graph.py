"""Graph data structures for time-series causal discovery.

The CDANs algorithm produces a graph that combines three kinds of structure:

1. **Lagged edges**: ``X_i[t-lag] -> X_j[t]`` for ``lag >= 1``. These are
   always directed (past causes present, never the reverse).
2. **Contemporaneous edges**: relationships between ``X_i[t]`` and ``X_j[t]``,
   either directed (``->``) or undirected (``--``) when not yet oriented.
3. **Surrogate edges**: ``C -> X_i[t]`` indicating ``X_i`` has a changing
   mechanism over the surrogate variable ``C`` (time index or domain id).

This module provides a single container that tracks all three.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class TimeSeriesGraph:
    """Causal graph over a time-series with surrogate variable.

    Edge encoding for the contemporaneous adjacency matrix ``contemp_adj``:

    * ``contemp_adj[i, j] == 1`` and ``contemp_adj[j, i] == 0``: ``X_i -> X_j``.
    * ``contemp_adj[i, j] == 1`` and ``contemp_adj[j, i] == 1``: ``X_i -- X_j``
      (undirected; orientation not yet determined).
    * ``contemp_adj[i, j] == 0`` and ``contemp_adj[j, i] == 0``: no edge.

    Lagged edges are stored as a set of ``(source, target, lag)`` tuples,
    always interpreted as directed ``X_source[t-lag] -> X_target[t]``.

    Parameters
    ----------
    n_vars:
        Number of observed variables (excluding the surrogate ``C``).
    tau_max:
        Maximum lag considered.
    """

    n_vars: int
    tau_max: int
    contemp_adj: np.ndarray = field(init=False)
    lagged_edges: set[tuple[int, int, int]] = field(default_factory=set)
    changing_modules: set[int] = field(default_factory=set)
    var_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.contemp_adj = np.zeros((self.n_vars, self.n_vars), dtype=np.int8)
        if self.var_names is not None and len(self.var_names) != self.n_vars:
            raise ValueError(
                f"var_names has length {len(self.var_names)} "
                f"but n_vars={self.n_vars}"
            )

    # --- contemporaneous edge manipulation ----------------------------------

    def add_contemp_undirected(self, i: int, j: int) -> None:
        """Add an undirected contemporaneous edge ``X_i -- X_j``."""
        self._check_var(i)
        self._check_var(j)
        if i == j:
            raise ValueError("self-loops are not allowed in contemporaneous graph")
        self.contemp_adj[i, j] = 1
        self.contemp_adj[j, i] = 1

    def remove_contemp_edge(self, i: int, j: int) -> None:
        """Remove a contemporaneous edge between ``X_i`` and ``X_j``."""
        self._check_var(i)
        self._check_var(j)
        self.contemp_adj[i, j] = 0
        self.contemp_adj[j, i] = 0

    def orient_contemp(self, src: int, dst: int) -> None:
        """Orient a contemporaneous edge as ``X_src -> X_dst``."""
        self._check_var(src)
        self._check_var(dst)
        if src == dst:
            raise ValueError("cannot orient self-loop")
        self.contemp_adj[src, dst] = 1
        self.contemp_adj[dst, src] = 0

    def has_contemp_edge(self, i: int, j: int) -> bool:
        return bool(self.contemp_adj[i, j] or self.contemp_adj[j, i])

    def is_contemp_directed(self, i: int, j: int) -> bool:
        """True iff there is a directed edge in either direction (not undirected)."""
        return bool(self.contemp_adj[i, j] != self.contemp_adj[j, i])

    def is_contemp_undirected(self, i: int, j: int) -> bool:
        return bool(self.contemp_adj[i, j] == 1 and self.contemp_adj[j, i] == 1)

    def contemp_neighbors(self, i: int) -> set[int]:
        """All variables connected to ``i`` by any contemporaneous edge."""
        self._check_var(i)
        nbrs = set()
        for j in range(self.n_vars):
            if j != i and self.has_contemp_edge(i, j):
                nbrs.add(j)
        return nbrs

    # --- lagged edge manipulation -------------------------------------------

    def add_lagged_edge(self, src: int, dst: int, lag: int) -> None:
        """Record a lagged edge ``X_src[t-lag] -> X_dst[t]``."""
        self._check_var(src)
        self._check_var(dst)
        if not 1 <= lag <= self.tau_max:
            raise ValueError(f"lag must be in [1, {self.tau_max}], got {lag}")
        self.lagged_edges.add((src, dst, lag))

    def lagged_parents(self, j: int) -> set[tuple[int, int]]:
        """Return ``{(source_var, lag)}`` for all lagged parents of ``X_j``."""
        self._check_var(j)
        return {(src, lag) for src, dst, lag in self.lagged_edges if dst == j}

    def lagged_parents_union(self, vars_: list[int] | set[int]) -> set[tuple[int, int]]:
        """Union of lagged parents across multiple targets."""
        out: set[tuple[int, int]] = set()
        for j in vars_:
            out |= self.lagged_parents(j)
        return out

    # --- changing modules ---------------------------------------------------

    def mark_changing(self, i: int) -> None:
        """Mark variable ``i`` as having a changing mechanism (``C -> X_i``)."""
        self._check_var(i)
        self.changing_modules.add(i)

    def unmark_changing(self, i: int) -> None:
        self.changing_modules.discard(i)

    # --- iteration / export -------------------------------------------------

    def directed_contemp_edges(self) -> Iterator[tuple[int, int]]:
        """Yield ``(src, dst)`` for every contemporaneous edge oriented as ``src -> dst``."""
        for i in range(self.n_vars):
            for j in range(self.n_vars):
                if i != j and self.contemp_adj[i, j] == 1 and self.contemp_adj[j, i] == 0:
                    yield i, j

    def undirected_contemp_edges(self) -> Iterator[tuple[int, int]]:
        """Yield ``(i, j)`` with ``i < j`` for each undirected contemporaneous edge."""
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                if self.is_contemp_undirected(i, j):
                    yield i, j

    def to_networkx(self) -> nx.DiGraph:
        """Export as a NetworkX directed graph.

        Nodes are tuples ``(var_index, "now")`` for current-time variables,
        ``(var_index, lag)`` for lagged variables (lag >= 1), and the string
        ``"C"`` for the surrogate node when there are changing modules.
        Undirected contemporaneous edges are emitted as a pair of directed edges.
        """
        g = nx.DiGraph()

        for v in range(self.n_vars):
            label = self.var_names[v] if self.var_names else f"X{v}"
            g.add_node((v, "now"), name=label, kind="contemporaneous")

        for src, dst, lag in self.lagged_edges:
            src_label = self.var_names[src] if self.var_names else f"X{src}"
            g.add_node((src, lag), name=f"{src_label}(t-{lag})", kind="lagged")
            g.add_edge((src, lag), (dst, "now"), kind="lagged", lag=lag)

        for i in range(self.n_vars):
            for j in range(self.n_vars):
                if i == j:
                    continue
                if self.contemp_adj[i, j] == 1:
                    g.add_edge((i, "now"), (j, "now"), kind="contemporaneous")

        if self.changing_modules:
            g.add_node("C", name="C", kind="surrogate")
            for v in self.changing_modules:
                g.add_edge("C", (v, "now"), kind="surrogate")

        return g

    # --- diagnostics --------------------------------------------------------

    def summary(self) -> str:
        """Render a human-readable summary."""
        names = self.var_names or [f"X{i}" for i in range(self.n_vars)]
        lines = [
            f"TimeSeriesGraph(n_vars={self.n_vars}, tau_max={self.tau_max})",
            f"  Lagged edges: {len(self.lagged_edges)}",
        ]
        for src, dst, lag in sorted(self.lagged_edges):
            lines.append(f"    {names[src]}(t-{lag}) -> {names[dst]}(t)")

        directed = list(self.directed_contemp_edges())
        undirected = list(self.undirected_contemp_edges())
        lines.append(f"  Contemporaneous edges: {len(directed)} directed, {len(undirected)} undirected")
        for src, dst in sorted(directed):
            lines.append(f"    {names[src]} -> {names[dst]}")
        for i, j in sorted(undirected):
            lines.append(f"    {names[i]} -- {names[j]}")

        lines.append(f"  Changing modules ({len(self.changing_modules)}):")
        if self.changing_modules:
            lines.append("    " + ", ".join(names[v] for v in sorted(self.changing_modules)))
        return "\n".join(lines)

    def _check_var(self, i: int) -> None:
        if not 0 <= i < self.n_vars:
            raise IndexError(f"variable index {i} out of range [0, {self.n_vars})")
