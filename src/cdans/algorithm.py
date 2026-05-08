"""Top-level CDANs estimator.

Provides a scikit-learn-style API around the four-step algorithm. Most users
should interact with the library through this class; the individual step
functions are exposed under :mod:`cdans.steps` for finer-grained control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from cdans.ci_tests import CITest
from cdans.graph.timeseries_graph import TimeSeriesGraph
from cdans.independent_change import AUTO, WidthSpec
from cdans.steps import (
    build_partial_graph,
    discover_lagged_adjacencies,
    orient_edges,
    refine_skeleton,
)


@dataclass
class CDANsResult:
    """Container for the output of :meth:`CDANs.fit`.

    Attributes
    ----------
    graph:
        The final :class:`TimeSeriesGraph` containing lagged edges,
        contemporaneous edges (oriented where possible), and changing modules.
    timings:
        Per-step wall-clock time in seconds.
    config:
        The configuration used to produce this result.
    """

    graph: TimeSeriesGraph
    timings: dict[str, float] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def lagged_edges(self) -> set[tuple[int, int, int]]:
        return self.graph.lagged_edges

    @property
    def contemporaneous_adjacency(self) -> np.ndarray:
        return self.graph.contemp_adj

    @property
    def changing_modules(self) -> set[int]:
        return self.graph.changing_modules

    def summary(self) -> str:
        parts = [self.graph.summary()]
        if self.timings:
            parts.append("Timings:")
            for k, v in self.timings.items():
                parts.append(f"  {k}: {v:.3f}s")
        return "\n".join(parts)


class CDANs:
    """Temporal causal discovery from autocorrelated, nonstationary time series.

    Implements the four-step algorithm from Ferdous, Hasan & Gani (2023):

    1. Identify lagged adjacencies via MCI tests.
    2. Build the partial undirected graph (lagged + contemporaneous + surrogate).
    3. Refine the contemporaneous skeleton and confirm changing modules using
       CI tests with lagged-parent conditioning sets.
    4. Orient contemporaneous edges via v-structures and Meek's rules.

    Parameters
    ----------
    tau_max:
        Maximum lag.
    alpha:
        Significance level for the final CI tests.
    pc_alpha:
        Significance level for the preliminary PC step inside Step 1
        (typically larger than ``alpha``).
    ci_test:
        CI test to use for Steps 1 and 3. Either a string
        (``"fisherz"`` or ``"kci"``) or an object implementing :class:`CITest`.
    surrogate:
        Either ``"time"`` (use the time index, default) or an array of shape
        ``(n_samples,)`` to use as the distribution-shift surrogate.
    max_extra_conds:
        Cap on contemporaneous neighbors added to the lagged conditioning set
        in Step 3.
    use_independent_change:
        Whether Step 4 should run the iterative independent-change-principle
        sink-finding sub-pass to orient undirected edges between two changing
        modules. Set to ``False`` to return only the Markov equivalence
        class.
    independent_change_width:
        Bandwidth for the kernels inside the independent-change score.
        ``"auto"`` (default) picks per-call via the median heuristic
        on the standardized parents — robust across datasets, fast.
        ``"gp"`` learns the bandwidth via Gaussian-process marginal-
        likelihood optimization with an ARD-RBF kernel — adaptive
        across DGPs but ~10–50× slower than ``"auto"`` per score call;
        recommended when the skeleton is recovered correctly but
        contemp orientations between changing modules look wrong.
        A positive float forces a manual bandwidth (``0.1`` reproduces
        the MATLAB reference's hard-coded default).
    verbose:
        Print per-step progress.

    Examples
    --------
    >>> from cdans import CDANs
    >>> from cdans.utils import generate_synthetic_cdans
    >>> ds = generate_synthetic_cdans(n_vars=4, n_samples=300, tau_max=2, seed=0)
    >>> model = CDANs(tau_max=2, alpha=0.05, ci_test="fisherz")
    >>> result = model.fit(ds.data)
    >>> print(result.summary())  # doctest: +SKIP
    """

    def __init__(
        self,
        tau_max: int = 2,
        alpha: float = 0.05,
        pc_alpha: float = 0.2,
        ci_test: str | CITest = "fisherz",
        surrogate: str | np.ndarray = "time",
        max_extra_conds: int = 2,
        use_independent_change: bool = True,
        independent_change_width: WidthSpec = AUTO,
        verbose: bool = False,
    ) -> None:
        if tau_max < 1:
            raise ValueError(f"tau_max must be >= 1, got {tau_max}")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0 < pc_alpha < 1:
            raise ValueError(f"pc_alpha must be in (0, 1), got {pc_alpha}")
        self.tau_max = tau_max
        self.alpha = alpha
        self.pc_alpha = pc_alpha
        self.ci_test = ci_test
        self.surrogate = surrogate
        self.max_extra_conds = max_extra_conds
        self.use_independent_change = use_independent_change
        self.independent_change_width = independent_change_width
        self.verbose = verbose

    def fit(
        self,
        data: np.ndarray,
        var_names: list[str] | None = None,
    ) -> CDANsResult:
        """Run the full four-step pipeline.

        Parameters
        ----------
        data:
            Time series, shape ``(n_samples, n_vars)``.
        var_names:
            Optional names for the variables (used in the result's summary).

        Returns
        -------
        CDANsResult
            The discovered graph plus diagnostics.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"data must be 2D, got shape {data.shape}")
        n_samples = data.shape[0]

        # Resolve surrogate
        if isinstance(self.surrogate, str):
            if self.surrogate == "time":
                surrogate_arr = np.arange(n_samples, dtype=float)
            else:
                raise ValueError(
                    f"surrogate string must be 'time', got {self.surrogate!r}"
                )
        else:
            surrogate_arr = np.asarray(self.surrogate, dtype=float).ravel()
            if surrogate_arr.shape[0] != n_samples:
                raise ValueError(
                    f"surrogate length {surrogate_arr.shape[0]} != n_samples {n_samples}"
                )

        timings: dict[str, float] = {}

        if self.verbose:
            print(f"[CDANs] Step 1: discovering lagged adjacencies (tau_max={self.tau_max})")
        t0 = perf_counter()
        graph = discover_lagged_adjacencies(
            data,
            tau_max=self.tau_max,
            ci_test=self.ci_test,
            alpha=self.alpha,
            pc_alpha=self.pc_alpha,
            var_names=var_names,
            verbose=self.verbose,
        )
        timings["step1_lagged"] = perf_counter() - t0

        if self.verbose:
            print(f"[CDANs] Step 2: building partial undirected graph")
        t0 = perf_counter()
        build_partial_graph(graph)
        timings["step2_partial"] = perf_counter() - t0

        if self.verbose:
            print(f"[CDANs] Step 3: refining skeleton with optimized CI tests")
        t0 = perf_counter()
        refine_skeleton(
            graph,
            data,
            surrogate=surrogate_arr,
            ci_test=self.ci_test,
            alpha=self.alpha,
            max_extra_conds=self.max_extra_conds,
            verbose=self.verbose,
        )
        timings["step3_skeleton"] = perf_counter() - t0

        if self.verbose:
            print(f"[CDANs] Step 4: orienting contemporaneous edges")
        t0 = perf_counter()
        orient_edges(
            graph,
            data=data,
            surrogate=surrogate_arr,
            use_independent_change=self.use_independent_change,
            independent_change_width=self.independent_change_width,
        )
        timings["step4_orient"] = perf_counter() - t0

        return CDANsResult(
            graph=graph,
            timings=timings,
            config={
                "tau_max": self.tau_max,
                "alpha": self.alpha,
                "pc_alpha": self.pc_alpha,
                "ci_test": getattr(self.ci_test, "name", str(self.ci_test)),
                "max_extra_conds": self.max_extra_conds,
            },
        )
