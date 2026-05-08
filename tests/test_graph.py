"""Tests for cdans.graph.* — graph data structure and Meek's rules."""

import numpy as np
import pytest

from cdans.graph import TimeSeriesGraph, apply_meek_rules


def test_graph_init():
    g = TimeSeriesGraph(n_vars=4, tau_max=2)
    assert g.n_vars == 4
    assert g.tau_max == 2
    assert g.contemp_adj.shape == (4, 4)
    assert np.all(g.contemp_adj == 0)
    assert g.lagged_edges == set()


def test_undirected_then_orient():
    g = TimeSeriesGraph(n_vars=3, tau_max=1)
    g.add_contemp_undirected(0, 1)
    assert g.is_contemp_undirected(0, 1)
    assert not g.is_contemp_directed(0, 1)
    g.orient_contemp(0, 1)
    assert g.is_contemp_directed(0, 1)
    assert not g.is_contemp_undirected(0, 1)


def test_lagged_edges():
    g = TimeSeriesGraph(n_vars=3, tau_max=2)
    g.add_lagged_edge(0, 1, 1)
    g.add_lagged_edge(2, 1, 2)
    assert g.lagged_parents(1) == {(0, 1), (2, 2)}
    assert g.lagged_parents(0) == set()


def test_lagged_edge_validation():
    g = TimeSeriesGraph(n_vars=2, tau_max=2)
    with pytest.raises(ValueError, match="lag"):
        g.add_lagged_edge(0, 1, 0)
    with pytest.raises(ValueError, match="lag"):
        g.add_lagged_edge(0, 1, 3)
    with pytest.raises(IndexError):
        g.add_lagged_edge(0, 5, 1)


def test_self_loop_rejected():
    g = TimeSeriesGraph(n_vars=2, tau_max=1)
    with pytest.raises(ValueError, match="self"):
        g.add_contemp_undirected(0, 0)
    with pytest.raises(ValueError, match="self"):
        g.orient_contemp(1, 1)


def test_summary_has_no_crashes():
    g = TimeSeriesGraph(n_vars=3, tau_max=2)
    g.add_lagged_edge(0, 1, 1)
    g.add_contemp_undirected(0, 2)
    g.orient_contemp(0, 2)
    g.mark_changing(2)
    text = g.summary()
    assert "Lagged edges" in text
    assert "Changing modules" in text


def test_to_networkx_roundtrip():
    g = TimeSeriesGraph(n_vars=3, tau_max=1)
    g.add_lagged_edge(0, 1, 1)
    g.add_contemp_undirected(0, 2)
    nx_graph = g.to_networkx()
    assert nx_graph.has_edge((0, 1), (1, "now"))
    # undirected contemporaneous => both directions
    assert nx_graph.has_edge((0, "now"), (2, "now"))
    assert nx_graph.has_edge((2, "now"), (0, "now"))


def test_meek_r1_orients_chain():
    # a -> b, b -- c, a not adj c  =>  b -> c
    g = TimeSeriesGraph(n_vars=3, tau_max=1)
    g.add_contemp_undirected(0, 1)
    g.orient_contemp(0, 1)
    g.add_contemp_undirected(1, 2)
    apply_meek_rules(g)
    assert g.is_contemp_directed(1, 2)
    # check direction explicitly
    assert g.contemp_adj[1, 2] == 1
    assert g.contemp_adj[2, 1] == 0


def test_meek_r2_no_cycle():
    # a -> b -> c, a -- c  =>  a -> c
    g = TimeSeriesGraph(n_vars=3, tau_max=1)
    g.add_contemp_undirected(0, 1)
    g.orient_contemp(0, 1)
    g.add_contemp_undirected(1, 2)
    g.orient_contemp(1, 2)
    g.add_contemp_undirected(0, 2)
    apply_meek_rules(g)
    assert g.contemp_adj[0, 2] == 1
    assert g.contemp_adj[2, 0] == 0


def test_meek_idempotent():
    g = TimeSeriesGraph(n_vars=4, tau_max=1)
    g.add_contemp_undirected(0, 1)
    g.orient_contemp(0, 1)
    g.add_contemp_undirected(1, 2)
    apply_meek_rules(g)
    snapshot = g.contemp_adj.copy()
    apply_meek_rules(g)
    np.testing.assert_array_equal(g.contemp_adj, snapshot)
