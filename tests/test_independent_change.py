"""Tests for cdans.independent_change and the Step 4 integration.

The discriminative tests pin down the algorithm's defining behavior:
the score for the correct causal direction (parents -> effect) should be
lower than the score for the reversed direction. They use synthetic
non-stationary data with a known mechanism.
"""

import numpy as np
import pytest

from cdans.graph import TimeSeriesGraph
from cdans.independent_change import (
    AUTO,
    _gram_to_kernel,
    _resolve_bandwidth,
    _standardize,
    independent_change_score,
)
from cdans.steps.step4_orient import _orient_independent_change, orient_edges


# ---------------------------------------------------------------------------
# Helpers (synthetic non-stationary DGPs)
# ---------------------------------------------------------------------------


def _make_xy_changing_mechanism(
    n: int = 300,
    seed: int = 0,
    drift_amplitude: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``X -> Y`` with a time-varying mechanism on Y.

    ``X[t] ~ N(0, 1)`` (stationary), and
    ``Y[t] = a(t) * X[t] + b(t) * eps``,
    where ``a(t)`` and ``b(t)`` drift with a smooth sinusoidal trend.
    Both X and Y are changing modules under this DGP (X via varying
    parameters affecting its conditional, Y via the same).

    Returns ``(X, Y, c_indx)``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    c = t.astype(float) / n
    a = 0.6 + drift_amplitude * np.sin(2 * np.pi * c)
    b = 0.4 + 0.2 * np.cos(2 * np.pi * c)

    # Make X depend on c so it is itself a changing module.
    x = (1.0 + 0.5 * np.sin(2 * np.pi * c + 1.0)) * rng.standard_normal(n)
    y = a * x + b * rng.standard_normal(n)
    return x.reshape(-1, 1), y.reshape(-1, 1), t.reshape(-1, 1).astype(float)


# ---------------------------------------------------------------------------
# Score function: math + invariants
# ---------------------------------------------------------------------------


def test_score_returns_finite_scalar():
    x, y, c = _make_xy_changing_mechanism(n=100, seed=0)
    s = independent_change_score(x, y, c)
    assert np.isfinite(s)
    assert isinstance(s, float)


def test_score_validation():
    x = np.zeros((10, 1))
    y = np.zeros((10, 1))
    c = np.zeros((10, 1))
    with pytest.raises(ValueError, match="width"):
        independent_change_score(x, y, c, width=0.0)
    with pytest.raises(ValueError, match="width_t"):
        independent_change_score(x, y, c, width_t=0.0)
    with pytest.raises(ValueError, match="auto.*gp.*positive float"):
        independent_change_score(x, y, c, width="not_auto")
    with pytest.raises(ValueError, match="sample-size mismatch"):
        independent_change_score(np.zeros((10, 1)), np.zeros((20, 1)), np.zeros((10, 1)))
    with pytest.raises(ValueError, match="T >= 5"):
        independent_change_score(np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1)))


# ---------------------------------------------------------------------------
# Auto bandwidth selection
# ---------------------------------------------------------------------------


def test_resolve_bandwidth_auto_returns_positive_finite():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 1))
    bw = _resolve_bandwidth("auto", X, name="width")
    assert bw > 0
    assert np.isfinite(bw)


def test_resolve_bandwidth_manual_passthrough():
    bw = _resolve_bandwidth(0.5, np.zeros((10, 1)), name="width")
    assert bw == 0.5
    bw_int = _resolve_bandwidth(2, np.zeros((10, 1)), name="width")
    assert bw_int == 2.0


def test_resolve_bandwidth_invalid_string_raises():
    with pytest.raises(ValueError, match="must be 'auto'"):
        _resolve_bandwidth("median", np.zeros((10, 1)), name="width")


def test_resolve_bandwidth_invalid_value_raises():
    for bad in [-1.0, 0.0, np.inf, np.nan]:
        with pytest.raises(ValueError):
            _resolve_bandwidth(bad, np.zeros((10, 1)), name="width")


def test_resolve_bandwidth_gp_inside_resolver_is_a_bug():
    """``"gp"`` is supposed to be intercepted at the score-function level
    and never reach _resolve_bandwidth — flag clearly if it does."""
    with pytest.raises(ValueError, match="bug"):
        _resolve_bandwidth("gp", np.zeros((10, 1)), name="width")


# ---------------------------------------------------------------------------
# GP-based bandwidth path
# ---------------------------------------------------------------------------


def test_fit_gp_bandwidths_returns_positive_finite():
    """The GP fit returns sensible bandwidths on standard synthetic data."""
    from cdans.independent_change import _fit_gp_bandwidths

    x, y, c = _make_xy_changing_mechanism(n=120, seed=0)
    x_std = _standardize(x)
    y_std = _standardize(y)
    c_std = _standardize(c)
    bw_x, bw_t = _fit_gp_bandwidths(x_std, y_std, c_std)
    assert np.isfinite(bw_x) and bw_x > 0
    assert np.isfinite(bw_t) and bw_t > 0
    # Clipping bounds inside the helper.
    assert 1e-2 <= bw_x <= 1e2
    assert 1e-2 <= bw_t <= 1e2


def test_score_with_gp_bandwidth_runs():
    x, y, c = _make_xy_changing_mechanism(n=150, seed=0)
    s = independent_change_score(x, y, c, width="gp")
    assert np.isfinite(s)


def test_gp_picks_correct_direction_majority_of_seeds():
    """GP isn't a strict improvement on every seed (it can over-fit at
    small T), but on this simple X -> Y synthetic it should be correct
    on a clear majority of seeds. Test that GP gets >= 6/10 seeds right —
    a low bar that still catches catastrophic regressions."""
    correct = 0
    for seed in range(10):
        x, y, c = _make_xy_changing_mechanism(n=300, seed=seed)
        s_xy = independent_change_score(x, y, c, width="gp")
        s_yx = independent_change_score(y, x, c, width="gp")
        if s_xy < s_yx:
            correct += 1
    assert correct >= 6, (
        f"GP path got only {correct}/10 seeds correct on the simple X->Y "
        "synthetic — likely a regression in _fit_gp_bandwidths or the "
        "score formula's GP path."
    )


def test_gp_overrides_width_t_silently():
    """When width='gp', width_t is ignored — both bandwidths come from
    the same GP fit. We check this by confirming that two different
    width_t settings produce the same score under width='gp'."""
    x, y, c = _make_xy_changing_mechanism(n=120, seed=0)
    s1 = independent_change_score(x, y, c, width="gp", width_t="auto")
    s2 = independent_change_score(x, y, c, width="gp", width_t=0.5)
    np.testing.assert_allclose(s1, s2, rtol=1e-12)


def test_gp_for_width_t_only_works():
    """The mixed combination (manual width, gp for width_t) should run."""
    x, y, c = _make_xy_changing_mechanism(n=120, seed=0)
    s = independent_change_score(x, y, c, width=0.1, width_t="gp")
    assert np.isfinite(s)


def test_gp_through_orient_edges_runs():
    """Top-level: the gp path is reachable through Step 4 / CDANs class."""
    from cdans.steps.step4_orient import orient_edges

    x, y, c = _make_xy_changing_mechanism(n=200, seed=0)
    data = np.column_stack([x.ravel(), y.ravel()])

    graph = TimeSeriesGraph(n_vars=2, tau_max=1)
    graph.add_contemp_undirected(0, 1)
    graph.mark_changing(0)
    graph.mark_changing(1)
    graph._witness_sets = {}  # type: ignore[attr-defined]

    orient_edges(
        graph,
        data=data,
        surrogate=c.ravel(),
        use_independent_change=True,
        independent_change_width="gp",
    )
    # Either oriented or not, but the call must succeed.
    assert (
        graph.is_contemp_directed(0, 1)
        or graph.is_contemp_directed(1, 0)
        or graph.is_contemp_undirected(0, 1)
    )


def test_score_with_auto_bandwidth_runs():
    """Default (``"auto"``) should produce a finite score on standard
    synthetic data."""
    x, y, c = _make_xy_changing_mechanism(n=200, seed=0)
    s = independent_change_score(x, y, c)  # all defaults = "auto"
    assert np.isfinite(s)


def test_auto_and_manual_agree_when_passed_same_value():
    """Resolving ``"auto"`` and computing the bandwidth manually with the
    same scale, then passing it as a float, should give identical scores."""
    from cdans.ci_tests._kernels import median_bandwidth
    from cdans.independent_change import _AUTO_WIDTH_XY_SCALE

    x, y, c = _make_xy_changing_mechanism(n=150, seed=2)
    # Match what the score function does internally: standardize first,
    # apply the scale factor for X/Y but not for the surrogate.
    x_std = _standardize(x)
    c_std = _standardize(c)
    bw_x = _AUTO_WIDTH_XY_SCALE * median_bandwidth(x_std)
    bw_t = median_bandwidth(c_std)

    s_auto = independent_change_score(x, y, c, width=AUTO, width_t=AUTO)
    s_manual = independent_change_score(x, y, c, width=bw_x, width_t=bw_t)
    np.testing.assert_allclose(s_auto, s_manual, rtol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_auto_bandwidth_picks_correct_direction(seed):
    """The defining property must hold under auto bandwidth too:
    score(X -> Y) < score(Y -> X) when X causes Y."""
    x, y, c = _make_xy_changing_mechanism(n=300, seed=seed)
    s_xy = independent_change_score(x, y, c)  # defaults: width=auto, width_t=auto
    s_yx = independent_change_score(y, x, c)
    assert s_xy < s_yx, (
        f"seed={seed}, auto bandwidths: "
        f"s(X->Y)={s_xy:.4e} should be < s(Y->X)={s_yx:.4e}"
    )


def test_score_handles_1d_inputs():
    x, y, c = _make_xy_changing_mechanism(n=80, seed=1)
    s_1d = independent_change_score(x.ravel(), y.ravel(), c.ravel())
    s_2d = independent_change_score(x, y, c)
    np.testing.assert_allclose(s_1d, s_2d, rtol=1e-12)


def test_score_handles_multi_dim_parents():
    rng = np.random.default_rng(0)
    n = 150
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 0.5 * x1 + 0.4 * x2 + 0.3 * rng.standard_normal(n)
    c = np.arange(n, dtype=float)

    parents = np.column_stack([x1, x2])
    s = independent_change_score(parents, y, c)
    assert np.isfinite(s)


# ---------------------------------------------------------------------------
# Score function: defining behavior (correct direction wins)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_correct_direction_has_lower_score(seed):
    """The defining property: score(X -> Y) < score(Y -> X) when X actually
    causes Y under a changing mechanism."""
    x, y, c = _make_xy_changing_mechanism(n=300, seed=seed)
    s_xy = independent_change_score(x, y, c, width=0.1)
    s_yx = independent_change_score(y, x, c, width=0.1)
    assert s_xy < s_yx, (
        f"seed={seed}: independent-change score should prefer X -> Y "
        f"(got s_xy={s_xy:.6e} >= s_yx={s_yx:.6e})"
    )


# ---------------------------------------------------------------------------
# Helpers: numerical primitives
# ---------------------------------------------------------------------------


def test_standardize_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3)) * 5 + 10
    Z = _standardize(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0, ddof=1), 1.0, atol=1e-12)


def test_standardize_constant_column_passthrough():
    X = np.column_stack([np.ones(20), np.arange(20.0)])
    Z = _standardize(X)
    # Constant column → mean-subtracted (=0), no NaNs.
    assert np.all(np.isfinite(Z))
    np.testing.assert_allclose(Z[:, 0], 0.0, atol=1e-12)


def test_gram_to_kernel_returns_psd_like_kernel():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((20, 20))
    M = A @ A.T  # PSD
    K = _gram_to_kernel(M)
    # Diagonal is 1 (squared distance to self is 0).
    np.testing.assert_allclose(np.diag(K), np.ones(20), atol=1e-12)
    # Off-diagonals in (0, 1].
    off = K[~np.eye(20, dtype=bool)]
    assert np.all(off > 0)
    assert np.all(off <= 1)


# ---------------------------------------------------------------------------
# Step 4 integration
# ---------------------------------------------------------------------------


def test_orient_independent_change_orients_known_xy_pair():
    """End-to-end on Step 4: build a graph with a single undirected edge
    between two changing modules generated with X -> Y, and confirm
    Step 4 orients it as X -> Y."""
    x, y, c = _make_xy_changing_mechanism(n=300, seed=0)
    data = np.column_stack([x.ravel(), y.ravel()])

    graph = TimeSeriesGraph(n_vars=2, tau_max=1)
    graph.add_contemp_undirected(0, 1)
    graph.mark_changing(0)
    graph.mark_changing(1)
    graph._witness_sets = {}  # type: ignore[attr-defined]

    orient_edges(
        graph,
        data=data,
        surrogate=c.ravel(),
        use_independent_change=True,
        independent_change_width=0.1,
    )

    assert graph.is_contemp_directed(0, 1)
    assert graph.contemp_adj[0, 1] == 1, "edge should point X -> Y"
    assert graph.contemp_adj[1, 0] == 0


def test_use_independent_change_false_leaves_pair_undirected():
    """With ``use_independent_change=False``, the same setup should leave
    the edge undirected (only Markov equivalence class is returned)."""
    x, y, c = _make_xy_changing_mechanism(n=200, seed=0)
    data = np.column_stack([x.ravel(), y.ravel()])

    graph = TimeSeriesGraph(n_vars=2, tau_max=1)
    graph.add_contemp_undirected(0, 1)
    graph.mark_changing(0)
    graph.mark_changing(1)
    graph._witness_sets = {}  # type: ignore[attr-defined]

    orient_edges(
        graph,
        data=data,
        surrogate=c.ravel(),
        use_independent_change=False,
    )

    assert graph.is_contemp_undirected(0, 1)


def test_sink_finding_skips_when_only_one_candidate():
    """If only one changing module has undirected edges, sink-finding
    should be a no-op (the loop's ``while len(candidates) > 1`` guard)."""
    x, y, c = _make_xy_changing_mechanism(n=100, seed=0)
    data = np.column_stack([x.ravel(), y.ravel()])

    graph = TimeSeriesGraph(n_vars=2, tau_max=1)
    graph.add_contemp_undirected(0, 1)
    # Only one of the two is marked changing.
    graph.mark_changing(0)

    _orient_independent_change(graph, data=data, surrogate=c.ravel(), width=0.1)
    # Should be untouched.
    assert graph.is_contemp_undirected(0, 1)


def test_orient_edges_skips_independent_change_without_data():
    """Without data/surrogate, Step 4 should still run (just skip the
    independent-change sub-pass)."""
    graph = TimeSeriesGraph(n_vars=3, tau_max=1)
    graph.add_contemp_undirected(0, 1)
    graph.add_contemp_undirected(1, 2)
    graph._witness_sets = {}  # type: ignore[attr-defined]

    # No raise, returns the same graph.
    out = orient_edges(graph, use_independent_change=True)
    assert out is graph
