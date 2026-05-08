"""Synthetic data generator for benchmarking CDANs.

Follows the data generation process described in CDANs (Ferdous et al., 2023),
Section 5.1.1 / Appendix C. The generator produces multivariate time series with:

* a known ground-truth causal graph spanning lagged and contemporaneous edges,
* configurable nonstationarity (a subset of variables have time-varying coefficients),
* configurable autocorrelation strength.

The produced data can be used for unit tests, benchmarking, and reproducing
the synthetic-data experiments from the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyntheticDataset:
    """Container for a synthetic dataset and its ground-truth structure.

    Attributes
    ----------
    data:
        Observations, shape ``(n_samples, n_vars)``.
    lagged_edges:
        Set of ground-truth lagged edges. Each tuple ``(i, j, lag)`` means
        ``X_i[t - lag] -> X_j[t]`` with ``lag >= 1``.
    contemporaneous_edges:
        Ground-truth contemporaneous DAG. Each tuple ``(i, j)`` means
        ``X_i[t] -> X_j[t]``.
    changing_modules:
        Indices of variables whose generating mechanism is nonstationary
        (time-varying coefficients).
    """

    data: np.ndarray
    lagged_edges: set[tuple[int, int, int]]
    contemporaneous_edges: set[tuple[int, int]]
    changing_modules: set[int]
    metadata: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_vars(self) -> int:
        return int(self.data.shape[1])


def generate_synthetic_cdans(
    n_vars: int = 4,
    n_samples: int = 500,
    tau_max: int = 2,
    n_changing: int = 2,
    autocorr: float = 0.4,
    contemp_strength: float = 0.5,
    lagged_strength: float = 0.4,
    noise_std: float = 0.5,
    nonstationary_amplitude: float = 0.6,
    seed: int | None = 42,
) -> SyntheticDataset:
    """Generate a synthetic nonstationary, autocorrelated time series.

    The data-generating process is, for each variable ``i`` and time ``t``::

        X_i[t] = a_ii(t) * X_i[t-1]                       # autoregressive term
                + sum_{(j,lag) in lagged_pa(i)}            # lagged parents
                      b_{ij,lag}(t) * X_j[t-lag]
                + sum_{j in contemp_pa(i)}                 # contemporaneous parents
                      c_{ij}(t) * X_j[t]
                + eps_i[t]

    For variables in ``changing_modules`` the coefficients ``a, b, c`` are
    smoothly varying functions of ``t`` (sinusoidal). For other variables the
    coefficients are constants.

    Parameters
    ----------
    n_vars:
        Number of observed variables.
    n_samples:
        Length of the generated time series.
    tau_max:
        Maximum lag for the random lagged-parent structure.
    n_changing:
        Number of variables whose mechanism is nonstationary.
    autocorr:
        Magnitude of the autoregressive coefficient ``a_ii``.
    contemp_strength:
        Magnitude of contemporaneous coefficients.
    lagged_strength:
        Magnitude of lagged coefficients.
    noise_std:
        Standard deviation of the additive noise term.
    nonstationary_amplitude:
        Amplitude of coefficient drift for changing modules. The effective
        coefficient at time ``t`` is ``base + amplitude * sin(2 * pi * t / T)``.
    seed:
        RNG seed for reproducibility. ``None`` for nondeterministic.

    Returns
    -------
    SyntheticDataset
        The data, ground-truth graph, and changing-module indices.
    """
    if n_changing > n_vars:
        raise ValueError(
            f"n_changing ({n_changing}) cannot exceed n_vars ({n_vars})"
        )
    if tau_max < 1:
        raise ValueError(f"tau_max must be >= 1, got {tau_max}")

    rng = np.random.default_rng(seed)

    # 1. Sample a contemporaneous DAG over n_vars (sparse, with topological order 0..n-1).
    contemp_edges: set[tuple[int, int]] = set()
    contemp_density = 0.4
    for j in range(n_vars):
        for i in range(j):  # i < j ensures DAG
            if rng.random() < contemp_density:
                contemp_edges.add((i, j))

    # 2. Sample lagged edges. Every variable gets its own AR(1) term plus a
    #    handful of cross-lagged parents.
    lagged_edges: set[tuple[int, int, int]] = set()
    for j in range(n_vars):
        lagged_edges.add((j, j, 1))  # autoregressive on every variable
        n_extra = rng.integers(1, max(2, n_vars // 2) + 1)
        for _ in range(int(n_extra)):
            i = int(rng.integers(0, n_vars))
            lag = int(rng.integers(1, tau_max + 1))
            if i == j and lag == 1:
                continue
            lagged_edges.add((i, j, lag))

    # 3. Pick which variables have changing modules.
    changing_modules: set[int] = set(
        rng.choice(n_vars, size=n_changing, replace=False).tolist()
    )

    # 4. Sample base coefficients for every edge.
    rng_choice_signs = lambda size: rng.choice([-1.0, 1.0], size=size)
    contemp_coef: dict[tuple[int, int], float] = {
        e: float(rng_choice_signs(()) * (contemp_strength + 0.1 * rng.standard_normal()))
        for e in contemp_edges
    }
    lagged_coef: dict[tuple[int, int, int], float] = {}
    for e in lagged_edges:
        i, j, _ = e
        if i == j:  # autoregressive
            lagged_coef[e] = float(autocorr)
        else:
            lagged_coef[e] = float(
                rng_choice_signs(()) * (lagged_strength + 0.1 * rng.standard_normal())
            )

    # 5. Phase offsets for the time-varying terms (so changing modules drift differently).
    phase = {i: float(rng.uniform(0, 2 * np.pi)) for i in changing_modules}

    def coef(base: float, var_index: int, t: int) -> float:
        """Modulate a base coefficient if ``var_index`` is a changing module."""
        if var_index not in changing_modules:
            return base
        drift = nonstationary_amplitude * np.sin(
            2.0 * np.pi * t / max(n_samples, 1) + phase[var_index]
        )
        return base * (1.0 + drift)

    # 6. Roll out the time series. We need to evaluate contemporaneous edges
    #    in topological order (0..n-1 by construction).
    data = np.zeros((n_samples, n_vars), dtype=float)
    # warm-up via random init for the first tau_max steps
    data[:tau_max] = rng.standard_normal((tau_max, n_vars)) * noise_std

    for t in range(tau_max, n_samples):
        for j in range(n_vars):  # topological order
            value = 0.0
            # lagged contributions
            for (src, dst, lag), base in lagged_coef.items():
                if dst != j:
                    continue
                value += coef(base, j, t) * data[t - lag, src]
            # contemporaneous contributions
            for (src, dst), base in contemp_coef.items():
                if dst != j:
                    continue
                value += coef(base, j, t) * data[t, src]
            # noise
            value += noise_std * rng.standard_normal()
            data[t, j] = value

    return SyntheticDataset(
        data=data,
        lagged_edges=lagged_edges,
        contemporaneous_edges=contemp_edges,
        changing_modules=changing_modules,
        metadata={
            "n_vars": n_vars,
            "n_samples": n_samples,
            "tau_max": tau_max,
            "seed": seed,
            "autocorr": autocorr,
        },
    )
