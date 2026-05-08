"""Step 1 of CDANs: identify lagged adjacencies via the PCMCI algorithm.

This module is a self-contained Python implementation of **PCMCI**
(Runge et al., 2019) — *not* PCMCI+. PCMCI+ extends PCMCI to also
discover contemporaneous edges in a single pass; CDANs handles
contemporaneous structure separately in Step 3 with a different
conditioning strategy that exploits the lagged parents discovered here.

There are no external causal-discovery dependencies: PC-stable parent
selection, MCI conditioning with proper time-shifting, and the final
test loop are all implemented here from scratch on top of NumPy plus
a pluggable :class:`~cdans.ci_tests.CITest`.

Algorithm overview
------------------

PCMCI is a two-phase procedure:

* **PC step (PC-stable, Colombo & Maathuis 2014).** For each target
  variable ``X_j[t]``, iteratively prune candidate lagged parents
  ``X_i[t-lag]`` by testing ``X_i[t-lag] ⊥ X_j[t] | S`` for conditioning
  sets ``S`` drawn from the strongest current candidates. Tested at
  the loose ``pc_alpha`` (typically ``> alpha``). The "stable" property
  (snapshotting the candidate ranking at the start of each cardinality
  iteration) makes the surviving set independent of the order in which
  variables are visited.

* **MCI step (Momentary Conditional Independence).** For each
  surviving candidate ``X_i[t-lag]``, run a final test conditioning on:

    1. the *other* lagged parents of ``X_j`` (no shift), and
    2. the lagged parents of the *source* ``X_i``, **time-shifted by
       ``lag``** so that they refer to time ``(t-lag) - lag'`` for
       parent offset ``lag'``.

  Tested at the strict ``alpha``. The shift in (2) is what makes MCI
  "momentary" — it conditions on the source's own past, not its future.

Reference
---------
Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., Sejdinovic, D. (2019).
*Detecting and quantifying causal associations in large nonlinear time
series datasets*. Science Advances 5, eaau4996.
"""

from __future__ import annotations

import numpy as np

from cdans.ci_tests import CITest, get_ci_test
from cdans.graph.timeseries_graph import TimeSeriesGraph
from cdans.utils.lagging import column_for, lagged_design_matrix


def discover_lagged_adjacencies(
    data: np.ndarray,
    tau_max: int,
    *,
    ci_test: str | CITest = "fisherz",
    alpha: float = 0.05,
    pc_alpha: float = 0.2,
    max_conds_dim: int | None = None,
    var_names: list[str] | None = None,
    verbose: bool = False,
) -> TimeSeriesGraph:
    """Identify lagged parents for every variable using PCMCI.

    Parameters
    ----------
    data:
        Time series, shape ``(n_samples, n_vars)``.
    tau_max:
        Maximum lag to consider.
    ci_test:
        CI test name (``"fisherz"`` or ``"kci"``) or an instance.
    alpha:
        Significance level for the final MCI step.
    pc_alpha:
        Looser significance level for the PC step. Following PCMCI
        conventions this is usually larger than ``alpha``.
    max_conds_dim:
        Cap on the size of conditioning sets in the PC step.
        ``None`` (default) means no cap.
    var_names:
        Optional variable names for the returned graph.
    verbose:
        Print per-variable progress.

    Returns
    -------
    TimeSeriesGraph
        Graph with ``lagged_edges`` populated and an empty
        contemporaneous skeleton (filled in by Step 2).
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0 < pc_alpha < 1:
        raise ValueError(f"pc_alpha must be in (0, 1), got {pc_alpha}")

    n_samples, n_vars = data.shape
    if n_samples <= tau_max + 5:
        raise ValueError(
            f"need n_samples > tau_max + 5, "
            f"got n_samples={n_samples}, tau_max={tau_max}"
        )

    ci = get_ci_test(ci_test)
    graph = TimeSeriesGraph(n_vars=n_vars, tau_max=tau_max, var_names=var_names)
    Y, X_lagged, col_index = lagged_design_matrix(data, tau_max)

    # PC step: per-target candidate selection.
    candidates: dict[int, list[int]] = {}
    for j in range(n_vars):
        cands = _pc_step(
            target=Y[:, j],
            X_lagged=X_lagged,
            n_lagged_cols=X_lagged.shape[1],
            ci_test=ci,
            pc_alpha=pc_alpha,
            max_conds_dim=max_conds_dim,
        )
        candidates[j] = cands
        if verbose:
            print(f"  [PCMCI/PC] target X{j}: {len(cands)} candidate lagged parents")

    # MCI step: confirm each candidate using the time-shifted parent set.
    _mci_step(
        graph=graph,
        Y=Y,
        X_lagged=X_lagged,
        candidates=candidates,
        col_index=col_index,
        n_vars=n_vars,
        tau_max=tau_max,
        ci_test=ci,
        alpha=alpha,
        verbose=verbose,
    )

    return graph


# --- PC step -----------------------------------------------------------------


def _pc_step(
    *,
    target: np.ndarray,
    X_lagged: np.ndarray,
    n_lagged_cols: int,
    ci_test: CITest,
    pc_alpha: float,
    max_conds_dim: int | None,
) -> list[int]:
    """PC-stable selection of candidate lagged parents.

    Returns the column indices in ``X_lagged`` that pass every CI test
    against ``target`` at level ``pc_alpha`` for conditioning sets up
    to ``max_conds_dim``.

    Following the PC-stable convention (Colombo & Maathuis 2014) used
    inside PCMCI, at each iteration the conditioning sets are drawn
    from the *strongest* surviving candidates, where strength is
    measured by the smallest p-value seen so far across tested subsets
    (a low p-value means strong evidence of dependence with the
    target). The ranking is snapshotted *before* the iteration starts
    so that within-iteration removals do not change the conditioning
    sets used for the remaining candidates.
    """
    candidates = list(range(n_lagged_cols))
    # min_pval[c] = smallest p-value the candidate has produced so far
    # across all tested conditioning sets (= strongest evidence of
    # dependence with the target).
    min_pval: dict[int, float] = {c: 1.0 for c in candidates}

    cond_size = 0
    while True:
        if max_conds_dim is not None and cond_size > max_conds_dim:
            break
        if len(candidates) <= cond_size:
            break

        # Snapshot ranking BEFORE this iteration begins (PC-stable).
        # Strongest first: smallest min_pval.
        ranked = sorted(candidates, key=lambda c: min_pval[c])
        to_remove: list[int] = []

        for c in ranked:
            other = [o for o in ranked if o != c]
            if len(other) < cond_size:
                continue
            cond_subset = other[:cond_size]  # top-N strongest others
            z = X_lagged[:, cond_subset] if cond_subset else None
            p = ci_test.pvalue(target, X_lagged[:, c], z)
            # Track strongest dependence evidence seen so far.
            if p < min_pval[c]:
                min_pval[c] = p
            if p > pc_alpha:
                to_remove.append(c)

        if not to_remove:
            break
        candidates = [c for c in candidates if c not in to_remove]
        cond_size += 1

    return candidates


# --- MCI step ----------------------------------------------------------------


def _mci_step(
    *,
    graph: TimeSeriesGraph,
    Y: np.ndarray,
    X_lagged: np.ndarray,
    candidates: dict[int, list[int]],
    col_index: list[tuple[int, int]],
    n_vars: int,
    tau_max: int,
    ci_test: CITest,
    alpha: float,
    verbose: bool,
) -> None:
    """Run the MCI step, mutating ``graph`` to record surviving lagged edges."""
    for j in range(n_vars):
        for c in candidates[j]:
            src_var, c_lag = col_index[c]
            cond_cols = _build_mci_conditioning(
                target=j,
                candidate_col=c,
                candidate_src=src_var,
                candidate_lag=c_lag,
                candidates=candidates,
                col_index=col_index,
                n_vars=n_vars,
                tau_max=tau_max,
            )
            cond_arr = X_lagged[:, sorted(cond_cols)] if cond_cols else None
            p = ci_test.pvalue(Y[:, j], X_lagged[:, c], cond_arr)
            if p <= alpha:
                graph.add_lagged_edge(src=src_var, dst=j, lag=c_lag)
        if verbose:
            n_kept = sum(1 for e in graph.lagged_edges if e[1] == j)
            print(f"  [PCMCI/MCI] target X{j}: {n_kept} lagged parents kept")


def _build_mci_conditioning(
    *,
    target: int,
    candidate_col: int,
    candidate_src: int,
    candidate_lag: int,
    candidates: dict[int, list[int]],
    col_index: list[tuple[int, int]],
    n_vars: int,
    tau_max: int,
) -> set[int]:
    """Build the MCI conditioning set for ``X_src[t-lag] -> X_target[t]``.

    The set comprises:

    1. The other candidate lagged parents of ``X_target`` (no shift).
    2. The candidate lagged parents of the source variable ``X_src``,
       **time-shifted by ``lag``**. A parent of ``X_src`` at offset
       ``lag'`` is at absolute time ``(t - lag) - lag'``, which in the
       lagged design matrix is the column for
       ``(parent_var, lag + lag')``. Parents whose shifted lag exceeds
       ``tau_max`` are dropped (we cannot represent them in the design
       matrix).

    The candidate column itself is excluded from the conditioning set.
    """
    cond_cols: set[int] = set(candidates[target]) - {candidate_col}

    for parent_col in candidates[candidate_src]:
        parent_var, parent_lag = col_index[parent_col]
        shifted_lag = parent_lag + candidate_lag
        if 1 <= shifted_lag <= tau_max:
            shifted_col = column_for(parent_var, shifted_lag, n_vars)
            cond_cols.add(shifted_col)

    cond_cols.discard(candidate_col)
    return cond_cols
