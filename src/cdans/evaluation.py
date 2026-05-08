"""Structure-recovery metrics for CDANs.

Compares a fitted :class:`TimeSeriesGraph` against a ground-truth one
(or a :class:`SyntheticDataset`) and reports standard causal-discovery
metrics for each edge type:

* **Lagged edges** — strict directed-edge comparison, since lagged
  edges always have a ``past -> present`` direction.
* **Contemporaneous edges** — PDAG-aware: for each unordered pair the
  prediction and truth fall into one of four states (no edge,
  ``i → j``, ``j → i``, undirected) and a disagreement is one SHD unit.
  Two flavours are reported:

    * **Skeleton metrics** — adjacency only, ignoring direction.
    * **Directed metrics** — exact directed-edge match (``i → j`` only
      counts as a true positive if the truth also has ``i → j``).
* **Changing modules** — set comparison on the set of variables
  receiving ``C → X_i``.

For each, :class:`GraphMetrics` exposes ``tp``/``fp``/``fn`` plus
derived rates: precision, recall (= TPR), FDR, FPR, F1.

The aggregate :class:`StructureRecoveryMetrics` collects everything in
a single object with a printable summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from cdans.graph.timeseries_graph import TimeSeriesGraph
from cdans.utils.synthetic import SyntheticDataset

TruthLike = Union[TimeSeriesGraph, SyntheticDataset]


# ---------------------------------------------------------------------------
# Per-edge-type metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphMetrics:
    """Edge-level confusion-matrix counts plus derived rates.

    Attributes
    ----------
    tp:
        True positives — edges in both prediction and truth.
    fp:
        False positives — edges in prediction but not in truth.
    fn:
        False negatives — edges in truth but not in prediction.
    n_truth:
        Total number of true edges (``tp + fn``). Convenience.
    n_pred:
        Total number of predicted edges (``tp + fp``). Convenience.
    """

    tp: int
    fp: int
    fn: int

    @property
    def n_truth(self) -> int:
        return self.tp + self.fn

    @property
    def n_pred(self) -> int:
        return self.tp + self.fp

    @property
    def precision(self) -> float:
        """``TP / (TP + FP)``. Returns ``0.0`` when no edges are predicted."""
        d = self.n_pred
        return self.tp / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        """``TP / (TP + FN)``. Same as :attr:`tpr`. Returns ``0.0`` when no
        true edges exist (i.e. truth is empty)."""
        d = self.n_truth
        return self.tp / d if d > 0 else 0.0

    @property
    def tpr(self) -> float:
        """True positive rate; alias for :attr:`recall`."""
        return self.recall

    @property
    def fdr(self) -> float:
        """False discovery rate ``= FP / (TP + FP) = 1 - precision``."""
        d = self.n_pred
        return self.fp / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall. ``0.0`` when both are zero."""
        p, r = self.precision, self.recall
        d = p + r
        return 2 * p * r / d if d > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"TP={self.tp:3d} FP={self.fp:3d} FN={self.fn:3d}  "
            f"P={self.precision:.2f} R={self.recall:.2f} "
            f"F1={self.f1:.2f} FDR={self.fdr:.2f}"
        )


def _metrics_from_sets(predicted: set, truth: set) -> GraphMetrics:
    """Confusion counts from two sets of comparable items."""
    return GraphMetrics(
        tp=len(predicted & truth),
        fp=len(predicted - truth),
        fn=len(truth - predicted),
    )


# ---------------------------------------------------------------------------
# Aggregate result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureRecoveryMetrics:
    """All metrics for a single graph-vs-truth comparison.

    Use :func:`evaluate_graph` to produce one of these.

    Attributes
    ----------
    lagged:
        :class:`GraphMetrics` for lagged edges (``src``, ``dst``, ``lag``).
    contemp_skeleton:
        :class:`GraphMetrics` for the contemporaneous skeleton (adjacency
        only — direction ignored).
    contemp_directed:
        :class:`GraphMetrics` for contemporaneous directed edges. An
        undirected predicted edge counts as neither TP nor FP for a
        directed truth edge; see also :attr:`shd_contemp` for a PDAG-aware
        single number.
    changing_modules:
        :class:`GraphMetrics` over the set of changing-module variable
        indices.
    shd_lagged:
        Structural Hamming Distance for lagged edges. Equal to the
        symmetric difference of the two edge sets.
    shd_contemp:
        PDAG-aware SHD for contemporaneous edges. For each unordered
        pair ``(i, j)``, the state is one of ``{no edge, i->j, j->i,
        undirected}``; one SHD unit is added per state mismatch.
    shd_total:
        ``shd_lagged + shd_contemp``. Changing-module disagreements are
        counted separately and not folded into total SHD.

    Notes
    -----
    Two convenience aggregates are also available as computed
    properties:

    * :attr:`total` — pools TP/FP/FN across :attr:`lagged`,
      :attr:`contemp_directed`, and :attr:`changing_modules` into a
      single :class:`GraphMetrics`. Use this when you want one overall
      TPR/FDR/F1 number per fit (the typical "full TPR/FDR" reported in
      benchmarks).
    * :attr:`total_skeleton` — same as ``total`` but uses
      :attr:`contemp_skeleton` instead of :attr:`contemp_directed`. More
      lenient: a contemp edge with the wrong direction still counts.
    """

    lagged: GraphMetrics
    contemp_skeleton: GraphMetrics
    contemp_directed: GraphMetrics
    changing_modules: GraphMetrics
    shd_lagged: int
    shd_contemp: int

    @property
    def shd_total(self) -> int:
        return self.shd_lagged + self.shd_contemp

    @property
    def total(self) -> GraphMetrics:
        """Aggregate metrics across all categories at the **directed-edge**
        level.

        Pools TP/FP/FN counts from :attr:`lagged`,
        :attr:`contemp_directed`, and :attr:`changing_modules` into a
        single :class:`GraphMetrics`. This yields the overall TPR, FDR,
        precision, recall, and F1 typically reported in causal-discovery
        papers as a single per-method number.

        Use :attr:`total_skeleton` for the more lenient adjacency-only
        version that doesn't penalize direction errors.
        """
        return GraphMetrics(
            tp=self.lagged.tp + self.contemp_directed.tp + self.changing_modules.tp,
            fp=self.lagged.fp + self.contemp_directed.fp + self.changing_modules.fp,
            fn=self.lagged.fn + self.contemp_directed.fn + self.changing_modules.fn,
        )

    @property
    def total_skeleton(self) -> GraphMetrics:
        """Aggregate metrics at the contemp-**skeleton** level (more lenient).

        Same as :attr:`total` but uses :attr:`contemp_skeleton` instead
        of :attr:`contemp_directed`, so a contemp edge with the wrong
        direction (or left undirected) still counts as a true positive
        as long as the adjacency was found.

        Useful for separating "did we find the right structure" from
        "did we orient it correctly".
        """
        return GraphMetrics(
            tp=self.lagged.tp + self.contemp_skeleton.tp + self.changing_modules.tp,
            fp=self.lagged.fp + self.contemp_skeleton.fp + self.changing_modules.fp,
            fn=self.lagged.fn + self.contemp_skeleton.fn + self.changing_modules.fn,
        )

    def summary(self) -> str:
        """Formatted multi-line summary."""
        lines = [
            "Structure recovery metrics",
            "=" * 60,
            f"  Lagged edges               {self.lagged}",
            f"  Contemp skeleton           {self.contemp_skeleton}",
            f"  Contemp directed           {self.contemp_directed}",
            f"  Changing modules           {self.changing_modules}",
            "  " + "-" * 58,
            f"  Total (directed)           {self.total}",
            f"  Total (skeleton)           {self.total_skeleton}",
            "",
            f"  SHD (lagged):    {self.shd_lagged}",
            f"  SHD (contemp):   {self.shd_contemp}",
            f"  SHD (total):     {self.shd_total}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def evaluate_graph(
    predicted: TimeSeriesGraph,
    truth: TruthLike,
) -> StructureRecoveryMetrics:
    """Compute structure-recovery metrics for a fitted graph.

    Parameters
    ----------
    predicted:
        The :class:`TimeSeriesGraph` returned by ``CDANs(...).fit(...)``
        (specifically ``result.graph``).
    truth:
        The ground-truth structure. Can be either a
        :class:`TimeSeriesGraph` (canonical) or a
        :class:`SyntheticDataset` (the convenience output of
        :func:`generate_synthetic_cdans`).

    Returns
    -------
    StructureRecoveryMetrics

    Raises
    ------
    ValueError
        If ``predicted`` and ``truth`` have inconsistent dimensions.

    Example
    -------
    >>> from cdans import CDANs
    >>> from cdans.utils import generate_synthetic_cdans
    >>> from cdans.evaluation import evaluate_graph
    >>> dataset = generate_synthetic_cdans(n_vars=4, n_samples=400, tau_max=2)
    >>> result = CDANs(tau_max=2).fit(dataset.data)
    >>> metrics = evaluate_graph(result.graph, dataset)
    >>> print(metrics.summary())  # doctest: +SKIP
    """
    truth_graph = _to_graph(truth)
    _check_consistent_dims(predicted, truth_graph)

    # Lagged edges: strict (src, dst, lag) match.
    lagged = _metrics_from_sets(
        set(predicted.lagged_edges),
        set(truth_graph.lagged_edges),
    )

    # Contemporaneous skeleton: undirected pairs where any edge exists.
    pred_skel = _undirected_pairs(predicted)
    truth_skel = _undirected_pairs(truth_graph)
    contemp_skeleton = _metrics_from_sets(pred_skel, truth_skel)

    # Contemporaneous directed: strict directed-edge match.
    pred_dir = set(predicted.directed_contemp_edges())
    truth_dir = set(truth_graph.directed_contemp_edges())
    contemp_directed = _metrics_from_sets(pred_dir, truth_dir)

    # Changing modules.
    changing = _metrics_from_sets(
        set(predicted.changing_modules),
        set(truth_graph.changing_modules),
    )

    return StructureRecoveryMetrics(
        lagged=lagged,
        contemp_skeleton=contemp_skeleton,
        contemp_directed=contemp_directed,
        changing_modules=changing,
        shd_lagged=_shd_lagged(predicted, truth_graph),
        shd_contemp=_shd_contemp(predicted, truth_graph),
    )


# ---------------------------------------------------------------------------
# Standalone metric helpers (also useful in isolation)
# ---------------------------------------------------------------------------


def shd(predicted: TimeSeriesGraph, truth: TruthLike) -> int:
    """Total Structural Hamming Distance (lagged + contemporaneous).

    Convenience wrapper around the corresponding fields of
    :func:`evaluate_graph`.
    """
    truth_graph = _to_graph(truth)
    _check_consistent_dims(predicted, truth_graph)
    return _shd_lagged(predicted, truth_graph) + _shd_contemp(
        predicted, truth_graph
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _to_graph(truth: TruthLike) -> TimeSeriesGraph:
    """Coerce a truth object to a :class:`TimeSeriesGraph`."""
    if isinstance(truth, TimeSeriesGraph):
        return truth
    if isinstance(truth, SyntheticDataset):
        # tau_max isn't a direct attribute of SyntheticDataset; the
        # generator stashes it in ``metadata``. Fall back to inferring
        # it from the lagged edges if metadata is absent.
        tau_max = truth.metadata.get("tau_max")
        if tau_max is None:
            tau_max = max(
                (lag for _, _, lag in truth.lagged_edges), default=1
            )
        g = TimeSeriesGraph(n_vars=truth.n_vars, tau_max=tau_max)
        for src, dst, lag in truth.lagged_edges:
            g.add_lagged_edge(src, dst, lag)
        for src, dst in truth.contemporaneous_edges:
            # SyntheticDataset stores all contemp edges as directed
            # (the generator knows the truth).
            g.orient_contemp(src, dst)
        for v in truth.changing_modules:
            g.mark_changing(v)
        return g
    raise TypeError(
        f"truth must be TimeSeriesGraph or SyntheticDataset, "
        f"got {type(truth).__name__}"
    )


def _check_consistent_dims(
    predicted: TimeSeriesGraph, truth: TimeSeriesGraph
) -> None:
    if predicted.n_vars != truth.n_vars:
        raise ValueError(
            f"predicted has n_vars={predicted.n_vars}, "
            f"truth has n_vars={truth.n_vars}"
        )
    if predicted.tau_max != truth.tau_max:
        raise ValueError(
            f"predicted has tau_max={predicted.tau_max}, "
            f"truth has tau_max={truth.tau_max}"
        )


def _undirected_pairs(graph: TimeSeriesGraph) -> set[tuple[int, int]]:
    """Set of unordered pairs ``(i, j)`` with ``i < j`` connected by any
    contemp edge (directed or undirected)."""
    n = graph.n_vars
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if graph.contemp_adj[i, j] == 1 or graph.contemp_adj[j, i] == 1:
                pairs.add((i, j))
    return pairs


def _shd_lagged(predicted: TimeSeriesGraph, truth: TimeSeriesGraph) -> int:
    """Lagged SHD = size of symmetric difference of edge sets."""
    return len(set(predicted.lagged_edges) ^ set(truth.lagged_edges))


def _shd_contemp(predicted: TimeSeriesGraph, truth: TimeSeriesGraph) -> int:
    """PDAG-aware contemporaneous SHD.

    For each unordered pair ``(i, j)``, classify into one of four states
    (no edge, ``i -> j``, ``j -> i``, undirected) and add 1 per state
    mismatch.
    """
    n = predicted.n_vars
    shd_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _contemp_pair_state(predicted, i, j) != _contemp_pair_state(
                truth, i, j
            ):
                shd_count += 1
    return shd_count


def _contemp_pair_state(graph: TimeSeriesGraph, i: int, j: int) -> str:
    """Return one of ``"none"``, ``"forward"`` (i->j), ``"reverse"`` (j->i),
    ``"undirected"``."""
    ij = int(graph.contemp_adj[i, j])
    ji = int(graph.contemp_adj[j, i])
    if ij == 0 and ji == 0:
        return "none"
    if ij == 1 and ji == 0:
        return "forward"
    if ij == 0 and ji == 1:
        return "reverse"
    return "undirected"


__all__ = [
    "GraphMetrics",
    "StructureRecoveryMetrics",
    "evaluate_graph",
    "shd",
]
