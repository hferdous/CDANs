"""End-to-end test of the CDANs pipeline on synthetic data.

This is the priority-1 deliverable: prove the library runs all four steps
on a synthetic dataset and recovers a meaningful causal structure.
"""

import numpy as np
import pytest

from cdans import CDANs
from cdans.utils import generate_synthetic_cdans


def _confusion(true_set: set, pred_set: set) -> tuple[int, int, int]:
    """Return (TP, FP, FN) of pred against true."""
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    return tp, fp, fn


def test_full_pipeline_runs_on_synthetic():
    """Smoke test: pipeline runs end to end and returns a populated graph."""
    ds = generate_synthetic_cdans(n_vars=4, n_samples=300, tau_max=2, seed=0)
    model = CDANs(tau_max=2, alpha=0.05, ci_test="fisherz")
    result = model.fit(ds.data)
    # Basic sanity
    assert result.graph.n_vars == 4
    assert result.graph.tau_max == 2
    # All four steps recorded a timing
    assert set(result.timings) == {
        "step1_lagged",
        "step2_partial",
        "step3_skeleton",
        "step4_orient",
    }
    # Should at least find the autoregressive lagged edges (every variable
    # has X_i[t-1] -> X_i[t] in the synthetic DGP)
    assert len(result.graph.lagged_edges) >= 1


def test_summary_renders():
    ds = generate_synthetic_cdans(n_vars=3, n_samples=200, tau_max=2, seed=1)
    result = CDANs(tau_max=2).fit(ds.data)
    text = result.summary()
    assert "TimeSeriesGraph" in text
    assert "Timings" in text


def test_lagged_recovery_quality():
    """Recall on lagged edges should be respectable on a clean synthetic DGP."""
    ds = generate_synthetic_cdans(
        n_vars=4,
        n_samples=600,
        tau_max=2,
        n_changing=0,  # stationary case is easier
        noise_std=0.3,
        seed=7,
    )
    result = CDANs(
        tau_max=2,
        alpha=0.05,
        pc_alpha=0.3,
        ci_test="fisherz",
    ).fit(ds.data)

    tp, fp, fn = _confusion(ds.lagged_edges, result.graph.lagged_edges)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    # On a 4-variable, lag-2, T=600 problem with linear-Gaussian DGP and
    # FisherZ, we should comfortably exceed 0.5 recall on lagged edges.
    assert recall > 0.5, (
        f"lagged recall too low: {recall:.2f} (TP={tp}, FP={fp}, FN={fn})"
    )
    # And we should not be wildly over-predicting (lots of false positives).
    assert precision > 0.2, (
        f"lagged precision too low: {precision:.2f} (TP={tp}, FP={fp}, FN={fn})"
    )


def test_changing_modules_are_subset_of_n_vars():
    ds = generate_synthetic_cdans(n_vars=5, n_samples=300, tau_max=2, seed=11)
    result = CDANs(tau_max=2).fit(ds.data)
    assert result.graph.changing_modules <= set(range(5))


def test_contemp_edges_are_within_skeleton():
    """No oriented contemp edge can exist between unconnected variables."""
    ds = generate_synthetic_cdans(n_vars=4, n_samples=250, tau_max=2, seed=13)
    result = CDANs(tau_max=2).fit(ds.data)
    n = result.graph.n_vars
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if result.graph.contemp_adj[i, j]:
                # there must be an underlying skeleton edge
                assert result.graph.has_contemp_edge(i, j)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pipeline_deterministic_for_given_seed(seed):
    ds = generate_synthetic_cdans(n_vars=4, n_samples=200, tau_max=2, seed=seed)
    r1 = CDANs(tau_max=2, ci_test="fisherz").fit(ds.data)
    r2 = CDANs(tau_max=2, ci_test="fisherz").fit(ds.data)
    assert r1.graph.lagged_edges == r2.graph.lagged_edges
    np.testing.assert_array_equal(r1.graph.contemp_adj, r2.graph.contemp_adj)
    assert r1.graph.changing_modules == r2.graph.changing_modules
