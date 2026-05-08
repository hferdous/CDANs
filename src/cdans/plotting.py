"""Plotting utilities for ``TimeSeriesGraph`` instances.

Two visualizations, both implemented as thin wrappers around
`tigramite.plotting`:

* :func:`plot_process_graph` — aggregate process view, one node per
  variable with curved arrows for lagged links and straight arrows for
  contemporaneous links.
* :func:`plot_time_series_graph` — time-unrolled view with a separate
  node per ``(variable, time-step)``; useful for inspecting which
  specific lag drives each edge.

Both functions accept a :class:`cdans.graph.TimeSeriesGraph` and convert
it to the ``(N, N, tau_max+1)`` string-link-matrix format that tigramite
uses internally. Tigramite and matplotlib are **optional** dependencies;
install with ``pip install "cdans[viz]"`` to enable plotting.

The surrogate variable ``C`` (changing-module driver) is materialized as
a virtual node ``X_N`` connected by ``C -> X_i`` arrows to every changing
module. This is the standard CD-NOD convention and renders cleanly in
the process graph; for time-series graphs, ``C`` is omitted by default
(it would add a column at every lag with no real meaning) and changing
modules are highlighted via node color instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from cdans.graph.timeseries_graph import TimeSeriesGraph


_TIGRAMITE_INSTALL_HINT = (
    'plotting requires tigramite and matplotlib. Install with: '
    'pip install "cdans[viz]"'
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_process_graph(
    graph: "TimeSeriesGraph",
    *,
    var_names: Sequence[str] | None = None,
    show_surrogate: bool = True,
    surrogate_name: str = "C",
    figsize: tuple[float, float] = (8.0, 6.0),
    save_path: str | None = None,
    fig_ax: tuple["Figure", "Axes"] | None = None,
    **tigramite_kwargs,
) -> tuple["Figure", "Axes"]:
    """Plot the aggregate process graph: one node per variable.

    Parameters
    ----------
    graph:
        The :class:`TimeSeriesGraph` to render.
    var_names:
        Optional list of length ``graph.n_vars``. Defaults to ``["X0",
        "X1", ...]``. The surrogate (if shown) is appended automatically.
    show_surrogate:
        If ``True`` (default) and the graph has at least one changing
        module, draw an extra node ``C`` with arrows ``C -> X_i`` to each
        changing module.
    surrogate_name:
        Label for the surrogate node. Default ``"C"``.
    figsize:
        Matplotlib figsize. Ignored if ``fig_ax`` is provided.
    save_path:
        If given, save the figure to this path with ``bbox_inches="tight"``
        at 150 dpi. The figure is also returned so the caller can do more.
    fig_ax:
        ``(fig, ax)`` to draw into. If ``None``, a new figure is created.
    **tigramite_kwargs:
        Forwarded to ``tigramite.plotting.plot_graph``. Useful examples:
        ``arrow_linewidth=4``, ``label_fontsize=12``, ``node_size=0.4``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    tp, plt = _import_plotting_deps()

    link_matrix, names = _to_link_matrix(
        graph,
        var_names=var_names,
        include_surrogate=show_surrogate,
        surrogate_name=surrogate_name,
    )

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_ax

    tp.plot_graph(
        graph=link_matrix,
        var_names=list(names),
        fig_ax=(fig, ax),
        **tigramite_kwargs,
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, ax


def plot_time_series_graph(
    graph: "TimeSeriesGraph",
    *,
    var_names: Sequence[str] | None = None,
    show_surrogate: bool = True,
    surrogate_name: str = "C",
    surrogate_position: str = "top",
    figsize: tuple[float, float] = (10.0, 5.0),
    save_path: str | None = None,
    fig_ax: tuple["Figure", "Axes"] | None = None,
    highlight_changing: bool = True,
    **tigramite_kwargs,
) -> tuple["Figure", "Axes"]:
    """Plot the time-unrolled causal graph.

    One node per ``(variable, time-step)``; lagged edges go between
    different time-steps, contemporaneous edges within the same one.

    Parameters
    ----------
    graph:
        The :class:`TimeSeriesGraph` to render.
    var_names:
        Optional list of length ``graph.n_vars``. Defaults to ``["X0",
        "X1", ...]``.
    show_surrogate:
        If ``True`` (default) and the graph has at least one changing
        module, include the surrogate ``C`` as an additional row with
        contemporaneous arrows ``C(t) -> X_i(t)`` to each changing
        module at every time-step. The surrogate's earlier-lag nodes
        are drawn for visual consistency but have no incoming or
        outgoing lagged edges.
    surrogate_name:
        Label for the surrogate row. Default ``"C"``.
    surrogate_position:
        ``"top"`` (default) or ``"bottom"``. Determines the surrogate
        row's vertical position relative to the data variables.
    figsize:
        Matplotlib figsize. Ignored if ``fig_ax`` is provided.
    save_path:
        If given, save the figure to this path with ``bbox_inches="tight"``
        at 150 dpi.
    fig_ax:
        ``(fig, ax)`` to draw into. If ``None``, a new figure is created.
    highlight_changing:
        If ``True``, color changing-module nodes differently from
        stationary ones (uses tigramite's ``special_nodes`` mechanism).
        Useful in addition to the explicit ``C -> X_i`` arrows because
        it makes the changing modules visually obvious at a glance.
    **tigramite_kwargs:
        Forwarded to ``tigramite.plotting.plot_time_series_graph``.
        Note that the ``order`` argument is set automatically based on
        ``surrogate_position``; passing it explicitly will raise.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    if "order" in tigramite_kwargs:
        raise TypeError(
            "the 'order' argument is set automatically based on "
            "surrogate_position; remove it from your call"
        )
    if surrogate_position not in ("top", "bottom"):
        raise ValueError(
            f"surrogate_position must be 'top' or 'bottom', got {surrogate_position!r}"
        )

    tp, plt = _import_plotting_deps()

    link_matrix, names = _to_link_matrix(
        graph,
        var_names=var_names,
        include_surrogate=show_surrogate,
        surrogate_name=surrogate_name,
    )

    n_vars = graph.n_vars
    has_surrogate = link_matrix.shape[0] == n_vars + 1
    n_total = link_matrix.shape[0]

    # Build the rendering order. Tigramite's `order` parameter is keyed
    # by variable index and gives the variable's vertical position
    # (0 = top, n_total - 1 = bottom): ``order[var_idx] = position``.
    # By default variables are rendered in their natural index order, so
    # without any rearrangement the surrogate (index n_vars, last) ends
    # up at the bottom. To put it at the top we shift every data
    # variable down by one and assign C to position 0.
    if has_surrogate and surrogate_position == "top":
        order = list(range(1, n_vars + 1)) + [0]
    else:
        order = list(range(n_total))

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_ax

    # Highlight changing modules via tigramite's special_nodes:
    # a dict {(var, lag): color} colours specific time-step nodes.
    special_nodes = None
    if highlight_changing and graph.changing_modules:
        special_nodes = {}
        for var in graph.changing_modules:
            for lag in range(graph.tau_max + 1):
                special_nodes[(var, -lag)] = "tab:orange"

    tp.plot_time_series_graph(
        graph=link_matrix,
        var_names=list(names),
        fig_ax=(fig, ax),
        special_nodes=special_nodes,
        order=order,
        **tigramite_kwargs,
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, ax


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _import_plotting_deps():
    """Import tigramite and matplotlib, with a clear error if missing."""
    try:
        from tigramite import plotting as tp  # type: ignore
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(_TIGRAMITE_INSTALL_HINT) from e
    return tp, plt


def _to_link_matrix(
    graph: "TimeSeriesGraph",
    *,
    var_names: Sequence[str] | None,
    include_surrogate: bool,
    surrogate_name: str = "C",
) -> tuple[np.ndarray, list[str]]:
    """Convert a TimeSeriesGraph to tigramite's ``(N, N, tau_max+1)`` array.

    The convention used by tigramite is:

    * ``'-->'`` for a directed edge ``i -> j``;
    * ``'<--'`` on the reverse cell, mirroring the forward cell;
    * ``'o-o'`` (with mirroring) for an undirected edge;
    * ``''`` when no edge is present.

    Lagged edges live in slices ``lag >= 1``; contemporaneous edges in
    slice ``0``.

    Returns
    -------
    link_matrix:
        Array of shape ``(N_total, N_total, tau_max+1)`` where
        ``N_total = graph.n_vars`` plus one if the surrogate is included.
    names:
        Variable names in matching order.
    """
    n = graph.n_vars
    if var_names is None:
        names = [f"X{i}" for i in range(n)]
    else:
        if len(var_names) != n:
            raise ValueError(
                f"var_names has length {len(var_names)}, expected {n}"
            )
        names = list(var_names)

    has_surrogate = include_surrogate and bool(graph.changing_modules)
    n_total = n + 1 if has_surrogate else n
    if has_surrogate:
        names = names + [surrogate_name]

    link = np.full((n_total, n_total, graph.tau_max + 1), "", dtype="<U3")

    # Lagged edges (always directed: src(t-lag) -> dst(t)).
    for src, dst, lag in graph.lagged_edges:
        link[src, dst, lag] = "-->"

    # Contemporaneous edges. The graph stores adjacency as
    # contemp_adj[i, j] == 1 iff there is an edge head/tail at j coming
    # from i. Symmetric pair == undirected; asymmetric == directed.
    for i in range(n):
        for j in range(i + 1, n):
            ij = int(graph.contemp_adj[i, j])
            ji = int(graph.contemp_adj[j, i])
            if ij == 1 and ji == 1:
                link[i, j, 0] = "o-o"
                link[j, i, 0] = "o-o"
            elif ij == 1 and ji == 0:
                link[i, j, 0] = "-->"
                link[j, i, 0] = "<--"
            elif ij == 0 and ji == 1:
                link[j, i, 0] = "-->"
                link[i, j, 0] = "<--"
            # else: no edge.

    # Surrogate -> changing-module edges (always contemporaneous, directed).
    if has_surrogate:
        c = n
        for cm in graph.changing_modules:
            link[c, cm, 0] = "-->"
            link[cm, c, 0] = "<--"

    return link, names


__all__ = ["plot_process_graph", "plot_time_series_graph"]
