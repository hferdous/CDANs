"""Plotting example: fit CDANs on synthetic data and render both graph views.

Run from the library root::

    pip install -e ".[viz]"
    python examples/plotting.py

Outputs ``process_graph.png`` and ``time_series_graph.png`` in the
current directory.
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from cdans import CDANs, plot_process_graph, plot_time_series_graph


def main() -> None:
    # Synthetic 5-variable lag-2 dataset with one changing module (X1).
    rng = np.random.default_rng(0)
    n = 400
    data = np.zeros((n, 5))
    data[:2] = 0.3 * rng.standard_normal((2, 5))
    for ti in range(2, n):
        cn = ti / n
        a = 0.5 + 0.3 * np.sin(2 * np.pi * cn)  # time-varying coefficient
        eps = 0.3 * rng.standard_normal(5)
        data[ti, 0] = a * data[ti - 1, 0] + eps[0]
        data[ti, 1] = 0.4 * data[ti - 1, 0] + 0.3 * data[ti - 2, 1] + eps[1]
        data[ti, 2] = a * data[ti - 1, 2] + eps[2]
        data[ti, 3] = 0.5 * data[ti, 2] + 0.3 * data[ti - 1, 3] + eps[3]
        data[ti, 4] = 0.3 * data[ti - 1, 4] + 0.4 * data[ti - 2, 3] + eps[4]

    print("Fitting CDANs...")
    result = CDANs(tau_max=2, alpha=0.05, ci_test="kci").fit(data)
    print(result.summary())

    var_names = ["HR", "BP", "SpO2", "RR", "Temp"]

    print("\nWriting process_graph.png ...")
    plot_process_graph(
        result.graph,
        var_names=var_names,
        save_path="process_graph.png",
        figsize=(8, 6),
    )

    print("Writing time_series_graph.png ...")
    plot_time_series_graph(
        result.graph,
        var_names=var_names,
        save_path="time_series_graph.png",
        figsize=(10, 5),
    )

    print("\nDone — check the two PNG files in this directory.")


if __name__ == "__main__":
    main()
