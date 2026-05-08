"""Tests for cdans.steps.step1_lagged — the PCMCI implementation.

Two things this file explicitly checks:

* The PC step is **PC-stable**: rerunning with shuffled column order
  produces the same surviving candidate set.
* The MCI step uses **time-shifted parents** of the source variable
  (the canonical MCI conditioning set from Runge et al., 2019), which
  is what distinguishes MCI from a plain conditional independence test.
"""

import numpy as np
import pytest

from cdans.ci_tests import FisherZ
from cdans.steps.step1_lagged import (
    _build_mci_conditioning,
    _pc_step,
    discover_lagged_adjacencies,
)
from cdans.utils import generate_synthetic_cdans
from cdans.utils.lagging import column_for, lagged_design_matrix


def test_mci_conditioning_includes_time_shifted_source_parents():
    """The MCI test for X_i[t-2] -> X_j[t] must condition on X_i's parents
    *at time t-2*, not at time t. In design-matrix terms, a parent of X_i
    at offset 1 must be included at lag=3 (= 2 + 1)."""
    n_vars = 3
    tau_max = 3
    # candidates: X_j (target=1) has X_i (var=0) at lag=2 as candidate;
    # X_i (target=0) has X_k (var=2) at lag=1 as candidate.
    col = lambda v, l: column_for(v, l, n_vars)
    candidates = {
        0: [col(2, 1)],   # X_i's parent: X_k at offset 1
        1: [col(0, 2)],   # X_j's parent: X_i at offset 2 (this is the candidate under test)
        2: [],
    }
    col_index = []
    for lag in range(1, tau_max + 1):
        for v in range(n_vars):
            col_index.append((v, lag))

    cond = _build_mci_conditioning(
        target=1,
        candidate_col=col(0, 2),
        candidate_src=0,
        candidate_lag=2,
        candidates=candidates,
        col_index=col_index,
        n_vars=n_vars,
        tau_max=tau_max,
    )
    # X_k at offset 1 is X_i's parent. Shifted by lag=2 of the candidate,
    # it becomes X_k at lag=3 from X_j's perspective.
    expected_shifted = col(2, 1 + 2)
    assert expected_shifted in cond, (
        f"MCI conditioning is missing the time-shifted source parent "
        f"(expected col {expected_shifted}, got {cond})"
    )
    # And it must NOT contain the un-shifted source parent:
    assert col(2, 1) not in cond, (
        "MCI conditioning incorrectly contains the un-shifted source parent"
    )


def test_mci_conditioning_drops_parents_beyond_tau_max():
    """If shifting a source's parent puts it beyond tau_max, it must be dropped."""
    n_vars = 2
    tau_max = 2
    col = lambda v, l: column_for(v, l, n_vars)
    candidates = {
        0: [col(1, 2)],   # X_0's parent: X_1 at offset 2
        1: [col(0, 2)],   # candidate under test: X_0 at offset 2 from X_1
    }
    col_index = []
    for lag in range(1, tau_max + 1):
        for v in range(n_vars):
            col_index.append((v, lag))

    cond = _build_mci_conditioning(
        target=1,
        candidate_col=col(0, 2),
        candidate_src=0,
        candidate_lag=2,
        candidates=candidates,
        col_index=col_index,
        n_vars=n_vars,
        tau_max=tau_max,
    )
    # X_1 at offset 2 shifted by lag=2 would be at lag=4 > tau_max=2,
    # so it must be dropped (no col_for(1, 4) call attempted).
    assert col(1, 2) not in cond  # un-shifted version excluded
    # The shifted version (lag=4) doesn't exist in the design matrix at all,
    # so no column index for it should appear.
    valid_cols = {col(v, l) for v in range(n_vars) for l in range(1, tau_max + 1)}
    assert cond.issubset(valid_cols)


def test_pc_step_is_pc_stable_under_column_reorder():
    """PC-stable means the surviving candidate set should not depend on
    the order in which candidates are visited. We verify by shuffling
    the design-matrix columns and checking we get the same survivors
    (modulo the permutation)."""
    rng = np.random.default_rng(0)
    n_samples = 400
    # Construct a deterministic linear-Gaussian DGP with known structure.
    eps = rng.standard_normal((n_samples, 3))
    x = np.zeros((n_samples, 3))
    for t in range(2, n_samples):
        x[t, 0] = 0.5 * x[t - 1, 0] + 0.4 * x[t - 1, 1] + 0.3 * eps[t, 0]
        x[t, 1] = 0.4 * x[t - 1, 1] + 0.3 * eps[t, 1]
        x[t, 2] = 0.4 * x[t - 1, 2] + 0.5 * x[t - 2, 0] + 0.3 * eps[t, 2]

    Y, X_lagged, _ = lagged_design_matrix(x, tau_max=2)
    target = Y[:, 2]  # X_2 has X_2[t-1] and X_0[t-2] as parents

    base = _pc_step(
        target=target,
        X_lagged=X_lagged,
        n_lagged_cols=X_lagged.shape[1],
        ci_test=FisherZ(),
        pc_alpha=0.2,
        max_conds_dim=None,
    )

    # Shuffle the columns and re-run.
    perm = rng.permutation(X_lagged.shape[1])
    inv = np.argsort(perm)
    shuffled = _pc_step(
        target=target,
        X_lagged=X_lagged[:, perm],
        n_lagged_cols=X_lagged.shape[1],
        ci_test=FisherZ(),
        pc_alpha=0.2,
        max_conds_dim=None,
    )
    # Translate shuffled column indices back to the original space.
    shuffled_in_orig = sorted(int(perm[i]) for i in shuffled)
    assert sorted(base) == shuffled_in_orig, (
        f"PC step is not PC-stable: base={sorted(base)}, "
        f"shuffled-back={shuffled_in_orig}"
    )


def test_step1_recovers_clear_lagged_structure():
    """Smoke test on a tiny dataset where the structure is unambiguous."""
    rng = np.random.default_rng(0)
    n = 600
    eps = rng.standard_normal((n, 3))
    x = np.zeros((n, 3))
    for t in range(2, n):
        x[t, 0] = 0.5 * x[t - 1, 0] + 0.3 * eps[t, 0]
        x[t, 1] = 0.4 * x[t - 1, 1] + 0.6 * x[t - 1, 0] + 0.3 * eps[t, 1]
        x[t, 2] = 0.4 * x[t - 1, 2] + 0.5 * x[t - 2, 1] + 0.3 * eps[t, 2]

    graph = discover_lagged_adjacencies(
        x, tau_max=2, ci_test="fisherz", alpha=0.05, pc_alpha=0.2,
    )
    # Expected lagged edges:
    #   (0, 0, 1), (1, 1, 1), (2, 2, 1)  -- self-AR
    #   (0, 1, 1)                         -- X_0[t-1] -> X_1[t]
    #   (1, 2, 2)                         -- X_1[t-2] -> X_2[t]
    must_have = {(0, 0, 1), (1, 1, 1), (2, 2, 1), (0, 1, 1), (1, 2, 2)}
    found = graph.lagged_edges
    missing = must_have - found
    assert not missing, f"PCMCI should recover {must_have} but missed {missing}"


def test_step1_runs_to_completion_on_synthetic():
    """End-to-end: runs to completion, returns a populated graph."""
    ds = generate_synthetic_cdans(n_vars=4, n_samples=300, tau_max=2, seed=0)
    graph = discover_lagged_adjacencies(ds.data, tau_max=2, ci_test="fisherz")
    assert graph.n_vars == 4
    assert graph.tau_max == 2
    assert isinstance(graph.lagged_edges, set)


def test_step1_too_short_data_raises():
    with pytest.raises(ValueError, match="n_samples"):
        discover_lagged_adjacencies(np.zeros((5, 3)), tau_max=2)


def test_step1_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        discover_lagged_adjacencies(np.zeros((100, 3)), tau_max=2, alpha=1.5)
    with pytest.raises(ValueError, match="pc_alpha"):
        discover_lagged_adjacencies(np.zeros((100, 3)), tau_max=2, pc_alpha=0.0)
