"""Tests for cdans.evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from cdans.evaluation import (
    GraphMetrics,
    StructureRecoveryMetrics,
    evaluate_graph,
    shd,
)
from cdans.graph import TimeSeriesGraph
from cdans.utils.synthetic import SyntheticDataset


# ---------------------------------------------------------------------------
# GraphMetrics arithmetic
# ---------------------------------------------------------------------------


def test_graph_metrics_perfect_match():
    m = GraphMetrics(tp=5, fp=0, fn=0)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.tpr == 1.0
    assert m.fdr == 0.0
    assert m.f1 == 1.0


def test_graph_metrics_all_wrong():
    """All predictions are FP, all truth is FN."""
    m = GraphMetrics(tp=0, fp=3, fn=4)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.fdr == 1.0
    assert m.f1 == 0.0


def test_graph_metrics_empty_predictions():
    """Predicted nothing, truth has 5 edges → recall=0, precision=0 by
    convention."""
    m = GraphMetrics(tp=0, fp=0, fn=5)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.fdr == 0.0
    assert m.f1 == 0.0
    assert m.n_truth == 5
    assert m.n_pred == 0


def test_graph_metrics_empty_truth():
    """Truth has 0 edges, predicted 3 → all FP → precision=0, recall=0
    by convention."""
    m = GraphMetrics(tp=0, fp=3, fn=0)
    assert m.precision == 0.0
    assert m.recall == 0.0  # division by 0 → 0.0
    assert m.fdr == 1.0
    assert m.n_truth == 0
    assert m.n_pred == 3


def test_graph_metrics_partial():
    m = GraphMetrics(tp=3, fp=2, fn=1)
    assert m.precision == pytest.approx(3 / 5)
    assert m.recall == pytest.approx(3 / 4)
    assert m.tpr == pytest.approx(3 / 4)
    assert m.fdr == pytest.approx(2 / 5)
    expected_f1 = 2 * (3 / 5) * (3 / 4) / ((3 / 5) + (3 / 4))
    assert m.f1 == pytest.approx(expected_f1)


def test_graph_metrics_str():
    m = GraphMetrics(tp=3, fp=2, fn=1)
    s = str(m)
    assert "TP=" in s and "FP=" in s and "FN=" in s
    assert "P=" in s and "R=" in s and "F1=" in s and "FDR=" in s


# ---------------------------------------------------------------------------
# evaluate_graph
# ---------------------------------------------------------------------------


def _make_truth_graph() -> TimeSeriesGraph:
    g = TimeSeriesGraph(n_vars=4, tau_max=2)
    g.add_lagged_edge(0, 1, lag=1)
    g.add_lagged_edge(2, 3, lag=2)
    g.add_lagged_edge(1, 1, lag=1)  # autoregressive
    g.orient_contemp(0, 2)         # X0 -> X2
    g.add_contemp_undirected(1, 3) # X1 -- X3
    g.mark_changing(0)
    g.mark_changing(2)
    return g


def test_evaluate_graph_perfect_recovery():
    truth = _make_truth_graph()
    # Predicted = identical to truth.
    pred = _make_truth_graph()
    metrics = evaluate_graph(pred, truth)

    assert metrics.lagged.tp == 3 and metrics.lagged.fp == 0 and metrics.lagged.fn == 0
    assert metrics.contemp_skeleton.tp == 2
    assert metrics.contemp_directed.tp == 1
    assert metrics.changing_modules.tp == 2

    assert metrics.shd_lagged == 0
    assert metrics.shd_contemp == 0
    assert metrics.shd_total == 0


def test_evaluate_graph_completely_empty_prediction():
    truth = _make_truth_graph()
    pred = TimeSeriesGraph(n_vars=4, tau_max=2)
    metrics = evaluate_graph(pred, truth)

    assert metrics.lagged.tp == 0 and metrics.lagged.fn == 3
    assert metrics.contemp_skeleton.fn == 2
    assert metrics.changing_modules.fn == 2

    assert metrics.shd_lagged == 3   # 3 missing lagged edges
    assert metrics.shd_contemp == 2  # 2 missing contemp pairs


def test_evaluate_graph_extra_edges_only():
    truth = TimeSeriesGraph(n_vars=3, tau_max=1)
    pred = TimeSeriesGraph(n_vars=3, tau_max=1)
    pred.add_lagged_edge(0, 1, lag=1)
    pred.orient_contemp(0, 2)
    pred.mark_changing(1)

    metrics = evaluate_graph(pred, truth)
    assert metrics.lagged.fp == 1 and metrics.lagged.tp == 0
    assert metrics.contemp_skeleton.fp == 1
    assert metrics.changing_modules.fp == 1
    assert metrics.shd_total == 2


def test_evaluate_graph_reversed_contemp_edge_counts_as_shd():
    """If truth says i -> j and we predict j -> i, that's one SHD unit."""
    truth = TimeSeriesGraph(n_vars=2, tau_max=1)
    truth.orient_contemp(0, 1)

    pred = TimeSeriesGraph(n_vars=2, tau_max=1)
    pred.orient_contemp(1, 0)  # reversed

    metrics = evaluate_graph(pred, truth)
    # Skeleton: same adjacency → TP, no FP/FN.
    assert metrics.contemp_skeleton.tp == 1
    assert metrics.contemp_skeleton.fp == 0
    assert metrics.contemp_skeleton.fn == 0
    # Directed: predicted (1,0) but truth (0,1) → both are wrong from
    # a directed perspective.
    assert metrics.contemp_directed.fp == 1
    assert metrics.contemp_directed.fn == 1
    # SHD: states differ → 1.
    assert metrics.shd_contemp == 1


def test_evaluate_graph_undirected_vs_directed_counts_as_shd():
    """Undirected predicted vs directed truth is a state mismatch."""
    truth = TimeSeriesGraph(n_vars=2, tau_max=1)
    truth.orient_contemp(0, 1)

    pred = TimeSeriesGraph(n_vars=2, tau_max=1)
    pred.add_contemp_undirected(0, 1)

    metrics = evaluate_graph(pred, truth)
    assert metrics.shd_contemp == 1
    # Skeleton: same adjacency.
    assert metrics.contemp_skeleton.tp == 1


def test_evaluate_graph_accepts_synthetic_dataset():
    """evaluate_graph should accept a SyntheticDataset as truth."""
    dataset = SyntheticDataset(
        data=np.zeros((10, 3)),
        lagged_edges={(0, 1, 1)},
        contemporaneous_edges={(0, 2)},
        changing_modules={0},
        metadata={"tau_max": 1},
    )
    pred = TimeSeriesGraph(n_vars=3, tau_max=1)
    pred.add_lagged_edge(0, 1, lag=1)
    pred.orient_contemp(0, 2)
    pred.mark_changing(0)

    metrics = evaluate_graph(pred, dataset)
    assert metrics.shd_total == 0
    assert metrics.lagged.tp == 1
    assert metrics.contemp_directed.tp == 1


def test_evaluate_graph_dimension_mismatch_raises():
    truth = TimeSeriesGraph(n_vars=4, tau_max=2)
    pred = TimeSeriesGraph(n_vars=3, tau_max=2)
    with pytest.raises(ValueError, match="n_vars"):
        evaluate_graph(pred, truth)

    pred2 = TimeSeriesGraph(n_vars=4, tau_max=1)
    with pytest.raises(ValueError, match="tau_max"):
        evaluate_graph(pred2, truth)


def test_evaluate_graph_invalid_truth_type():
    pred = TimeSeriesGraph(n_vars=2, tau_max=1)
    with pytest.raises(TypeError, match="TimeSeriesGraph or SyntheticDataset"):
        evaluate_graph(pred, "not a graph")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Convenience shd()
# ---------------------------------------------------------------------------


def test_shd_function_matches_evaluate_graph():
    truth = _make_truth_graph()
    pred = TimeSeriesGraph(n_vars=4, tau_max=2)
    pred.add_lagged_edge(0, 1, lag=1)  # match
    pred.add_lagged_edge(0, 3, lag=1)  # extra
    # missing (2, 3, 2) and (1, 1, 1)
    # missing all contemp

    quick_shd = shd(pred, truth)
    full = evaluate_graph(pred, truth)
    assert quick_shd == full.shd_total


# ---------------------------------------------------------------------------
# StructureRecoveryMetrics summary
# ---------------------------------------------------------------------------


def test_summary_contains_all_sections():
    metrics = StructureRecoveryMetrics(
        lagged=GraphMetrics(1, 0, 0),
        contemp_skeleton=GraphMetrics(0, 1, 0),
        contemp_directed=GraphMetrics(0, 0, 1),
        changing_modules=GraphMetrics(2, 0, 0),
        shd_lagged=0,
        shd_contemp=2,
    )
    summary = metrics.summary()
    assert "Lagged edges" in summary
    assert "Contemp skeleton" in summary
    assert "Contemp directed" in summary
    assert "Changing modules" in summary
    assert "Total (directed)" in summary
    assert "Total (skeleton)" in summary
    assert "SHD (total):" in summary
    assert metrics.shd_total == 2


# ---------------------------------------------------------------------------
# Total aggregates
# ---------------------------------------------------------------------------


def test_total_aggregates_directed_level():
    """total pools lagged + contemp_directed + changing_modules counts."""
    metrics = StructureRecoveryMetrics(
        lagged=GraphMetrics(tp=5, fp=3, fn=1),
        contemp_skeleton=GraphMetrics(tp=2, fp=1, fn=1),  # not used in total
        contemp_directed=GraphMetrics(tp=1, fp=2, fn=2),
        changing_modules=GraphMetrics(tp=2, fp=0, fn=1),
        shd_lagged=0,
        shd_contemp=0,
    )
    total = metrics.total
    assert total.tp == 5 + 1 + 2  # 8
    assert total.fp == 3 + 2 + 0  # 5
    assert total.fn == 1 + 2 + 1  # 4
    assert total.tpr == pytest.approx(8 / 12)
    assert total.fdr == pytest.approx(5 / 13)


def test_total_skeleton_uses_contemp_skeleton_instead():
    """total_skeleton uses contemp_skeleton in place of contemp_directed."""
    metrics = StructureRecoveryMetrics(
        lagged=GraphMetrics(tp=5, fp=3, fn=1),
        contemp_skeleton=GraphMetrics(tp=3, fp=1, fn=0),  # used here
        contemp_directed=GraphMetrics(tp=1, fp=2, fn=2),  # ignored here
        changing_modules=GraphMetrics(tp=2, fp=0, fn=1),
        shd_lagged=0,
        shd_contemp=0,
    )
    total_sk = metrics.total_skeleton
    assert total_sk.tp == 5 + 3 + 2  # 10
    assert total_sk.fp == 3 + 1 + 0  # 4
    assert total_sk.fn == 1 + 0 + 1  # 2


def test_total_perfect_recovery_gives_perfect_aggregate():
    truth = _make_truth_graph()
    metrics = evaluate_graph(_make_truth_graph(), truth)
    assert metrics.total.tpr == 1.0
    assert metrics.total.fdr == 0.0
    assert metrics.total.f1 == 1.0
    assert metrics.total_skeleton.tpr == 1.0


def test_total_is_at_least_as_good_as_directed():
    """total_skeleton.tpr >= total.tpr, since skeleton matching is more
    lenient."""
    truth = _make_truth_graph()
    pred = TimeSeriesGraph(n_vars=4, tau_max=2)
    pred.add_lagged_edge(0, 1, lag=1)
    pred.orient_contemp(2, 0)  # reversed direction (truth: 0 -> 2)
    pred.add_contemp_undirected(1, 3)  # matches truth's undirected
    pred.mark_changing(0)

    metrics = evaluate_graph(pred, truth)
    # Skeleton aggregate should have >= TP than directed aggregate.
    assert metrics.total_skeleton.tp >= metrics.total.tp
