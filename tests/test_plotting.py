"""Tests for cdans.plotting.

Tigramite is an optional dependency, so plotting tests are skipped when
it isn't installed. The :func:`_to_link_matrix` conversion is tested
unconditionally — it's pure NumPy and doesn't need tigramite.
"""

from __future__ import annotations

import numpy as np
import pytest

from cdans.graph import TimeSeriesGraph
from cdans.plotting import _to_link_matrix

# Optional plotting deps.
tigramite = pytest.importorskip("tigramite", reason="tigramite not installed")
matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")  # headless backend for tests


# ---------------------------------------------------------------------------
# _to_link_matrix: pure NumPy logic
# ---------------------------------------------------------------------------


def _toy_graph() -> TimeSeriesGraph:
    """A 3-var lag-1 graph with mixed structure: lagged, directed contemp,
    undirected contemp, and one changing module."""
    g = TimeSeriesGraph(n_vars=3, tau_max=1)
    g.add_lagged_edge(0, 1, lag=1)        # X0(t-1) -> X1(t)
    g.orient_contemp(0, 2)                # X0 -> X2 (directed)
    g.add_contemp_undirected(1, 2)        # X1 -- X2 (undirected)
    g.mark_changing(0)                    # X0 is a changing module
    return g


def test_link_matrix_shape_without_surrogate():
    g = _toy_graph()
    link, names = _to_link_matrix(g, var_names=None, include_surrogate=False)
    assert link.shape == (3, 3, 2)
    assert names == ["X0", "X1", "X2"]


def test_link_matrix_shape_with_surrogate():
    g = _toy_graph()
    link, names = _to_link_matrix(g, var_names=None, include_surrogate=True)
    assert link.shape == (4, 4, 2)
    assert names == ["X0", "X1", "X2", "C"]


def test_link_matrix_no_surrogate_when_no_changing_modules():
    g = TimeSeriesGraph(n_vars=2, tau_max=1)
    g.add_contemp_undirected(0, 1)
    link, names = _to_link_matrix(g, var_names=None, include_surrogate=True)
    assert link.shape == (2, 2, 2), "no surrogate node should be added"
    assert names == ["X0", "X1"]


def test_link_matrix_lagged_edge_encoding():
    g = _toy_graph()
    link, _ = _to_link_matrix(g, var_names=None, include_surrogate=False)
    assert link[0, 1, 1] == "-->"
    # Only the forward direction is set for lagged edges.
    assert link[1, 0, 1] == ""


def test_link_matrix_directed_contemp_encoding():
    g = _toy_graph()
    link, _ = _to_link_matrix(g, var_names=None, include_surrogate=False)
    # X0 -> X2 contemp: forward "-->" and reverse "<--"
    assert link[0, 2, 0] == "-->"
    assert link[2, 0, 0] == "<--"


def test_link_matrix_undirected_contemp_encoding():
    g = _toy_graph()
    link, _ = _to_link_matrix(g, var_names=None, include_surrogate=False)
    # X1 -- X2 undirected: "o-o" on both sides
    assert link[1, 2, 0] == "o-o"
    assert link[2, 1, 0] == "o-o"


def test_link_matrix_surrogate_edges():
    g = _toy_graph()
    link, names = _to_link_matrix(g, var_names=None, include_surrogate=True)
    c = names.index("C")
    # X0 is the only changing module; C -> X0 contemporaneously.
    assert link[c, 0, 0] == "-->"
    assert link[0, c, 0] == "<--"
    # Non-changing variables get no surrogate edge.
    assert link[c, 1, 0] == ""
    assert link[c, 2, 0] == ""


def test_link_matrix_custom_var_names():
    g = _toy_graph()
    link, names = _to_link_matrix(
        g, var_names=["alpha", "beta", "gamma"], include_surrogate=False,
    )
    assert names == ["alpha", "beta", "gamma"]


def test_link_matrix_var_names_length_validation():
    g = _toy_graph()
    with pytest.raises(ValueError, match="var_names has length"):
        _to_link_matrix(g, var_names=["only_one"], include_surrogate=False)


def test_link_matrix_custom_surrogate_name():
    g = _toy_graph()
    _, names = _to_link_matrix(
        g, var_names=None, include_surrogate=True, surrogate_name="time",
    )
    assert names[-1] == "time"


# ---------------------------------------------------------------------------
# Plot functions: smoke tests (just check they produce a Figure)
# ---------------------------------------------------------------------------


def test_plot_process_graph_returns_figure_axes(tmp_path):
    from cdans.plotting import plot_process_graph

    g = _toy_graph()
    fig, ax = plot_process_graph(
        g, save_path=str(tmp_path / "process.png"), figsize=(4, 3),
    )
    assert fig is not None
    assert ax is not None
    assert (tmp_path / "process.png").exists()


def test_plot_time_series_graph_returns_figure_axes(tmp_path):
    from cdans.plotting import plot_time_series_graph

    g = _toy_graph()
    fig, ax = plot_time_series_graph(
        g, save_path=str(tmp_path / "ts.png"), figsize=(6, 3),
    )
    assert fig is not None
    assert ax is not None
    assert (tmp_path / "ts.png").exists()


def test_plot_process_graph_no_changing_modules(tmp_path):
    """Graph without changing modules should still plot (no surrogate node)."""
    from cdans.plotting import plot_process_graph

    g = TimeSeriesGraph(n_vars=2, tau_max=1)
    g.add_lagged_edge(0, 1, lag=1)
    fig, _ = plot_process_graph(g, save_path=str(tmp_path / "p.png"), figsize=(4, 3))
    assert fig is not None


def test_plot_time_series_graph_highlights_changing(tmp_path):
    """Highlighting changing modules shouldn't blow up on edge cases."""
    from cdans.plotting import plot_time_series_graph

    g = _toy_graph()
    fig, _ = plot_time_series_graph(
        g,
        save_path=str(tmp_path / "ts_h.png"),
        figsize=(6, 3),
        highlight_changing=True,
    )
    assert fig is not None

    fig2, _ = plot_time_series_graph(
        g,
        save_path=str(tmp_path / "ts_nh.png"),
        figsize=(6, 3),
        highlight_changing=False,
    )
    assert fig2 is not None


def test_plot_time_series_graph_surrogate_at_top_by_default(tmp_path):
    """C should be drawn in the time-unrolled view by default, at top."""
    from cdans.plotting import plot_time_series_graph

    g = _toy_graph()
    # Default (show_surrogate=True, surrogate_position='top') should run.
    fig, _ = plot_time_series_graph(
        g, save_path=str(tmp_path / "ts_top.png"), figsize=(6, 3),
    )
    assert fig is not None


def test_plot_time_series_graph_surrogate_position_options(tmp_path):
    """All three surrogate-position settings should render without error."""
    from cdans.plotting import plot_time_series_graph

    g = _toy_graph()
    for kwargs in (
        {"show_surrogate": True, "surrogate_position": "top"},
        {"show_surrogate": True, "surrogate_position": "bottom"},
        {"show_surrogate": False},
    ):
        fig, _ = plot_time_series_graph(
            g, figsize=(6, 3), **kwargs,
        )
        assert fig is not None


def test_plot_time_series_graph_invalid_surrogate_position():
    from cdans.plotting import plot_time_series_graph

    g = _toy_graph()
    with pytest.raises(ValueError, match="surrogate_position"):
        plot_time_series_graph(g, surrogate_position="middle")


def test_plot_time_series_graph_rejects_explicit_order_kwarg():
    """The 'order' kwarg conflicts with surrogate_position; reject it."""
    from cdans.plotting import plot_time_series_graph

    g = _toy_graph()
    with pytest.raises(TypeError, match="order"):
        plot_time_series_graph(g, order=[0, 1, 2])
