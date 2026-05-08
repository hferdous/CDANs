"""How to run CDANs on a CSV file with custom variable names.

This example is self-contained: it generates a small synthetic CSV first,
then walks through loading it with pandas and fitting CDANs while
preserving the column names. To use it on your own data, just replace
the CSV-generation section with a path to your file.

Required CSV format
-------------------
* Rows are time-ordered samples (oldest to newest, top to bottom).
* Columns are variables. Each column is one variable's time series.
* No missing values. Drop or impute NaNs before passing to CDANs.
* If your CSV has a timestamp column, drop it before fitting — CDANs
  uses row order as the time axis. (You can pass an explicit
  ``surrogate=`` array to ``CDANs(...)`` if you need a non-time index.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cdans import CDANs, evaluate_graph
from cdans.utils import generate_synthetic_cdans


def write_demo_csv(path: Path) -> None:
    """Generate a small CSV so this example is runnable as-is.

    Skip this whole function when adapting to your own data — replace
    ``path`` below with the path to your real CSV.
    """
    dataset = generate_synthetic_cdans(
        n_vars=4, n_samples=800, tau_max=1, n_changing=1, seed=7,
    )
    df = pd.DataFrame(
        dataset.data,
        columns=["systolic_bp", "heart_rate", "spo2", "temperature"],
    )
    df.to_csv(path, index=False)
    print(f"Wrote demo CSV: {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")


def main() -> None:
    csv_path = Path("/tmp/demo_timeseries.csv")
    write_demo_csv(csv_path)

    # --- Load the CSV. -----------------------------------------------------
    # Replace this with `df = pd.read_csv("path/to/your/data.csv")`.
    # If your CSV has a timestamp column, drop it before fitting:
    #     df = pd.read_csv("...").drop(columns=["timestamp"])
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {df.shape[0]} samples × {df.shape[1]} variables")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:\n{df.head(3)}")

    # --- Sanity checks. ----------------------------------------------------
    if df.isna().any().any():
        raise ValueError(
            "CSV contains NaNs. Drop or impute them before fitting.\n"
            f"NaN counts per column:\n{df.isna().sum()}"
        )
    if not all(np.issubdtype(t, np.number) for t in df.dtypes):
        raise ValueError(
            "All columns must be numeric. Drop or encode any text columns."
        )

    # --- Fit CDANs. --------------------------------------------------------
    # Pass df.values for the data and df.columns.tolist() for the names.
    # The graph stores the names and uses them in summary() and plotting.
    model = CDANs(
        tau_max=1,           # max lag to consider; tune for your data
        alpha=0.05,          # CI-test significance threshold
        ci_test="kci",       # use "fisherz" if you want linear-only and faster
    )
    result = model.fit(df.values, var_names=df.columns.tolist())

    print(f"\nRuntime: {sum(result.timings.values()):.1f}s")

    # --- Inspect the recovered graph. --------------------------------------
    print("\nRecovered graph:")
    print(result.graph.summary())

    # --- Extract specific findings programmatically. -----------------------
    g = result.graph
    print("\nLagged edges (by name):")
    for src_idx, dst_idx, lag in sorted(g.lagged_edges):
        print(f"  {df.columns[src_idx]}[t-{lag}] -> {df.columns[dst_idx]}[t]")

    print("\nContemporaneous directed edges (by name):")
    for src_idx, dst_idx in sorted(g.directed_contemp_edges()):
        print(f"  {df.columns[src_idx]} -> {df.columns[dst_idx]}")

    print("\nContemporaneous undirected edges (by name):")
    for i, j in sorted(g.undirected_contemp_edges()):
        print(f"  {df.columns[i]} -- {df.columns[j]}")

    print("\nVariables with non-stationary mechanisms (changing modules):")
    for v in sorted(g.changing_modules):
        print(f"  {df.columns[v]}")


if __name__ == "__main__":
    main()
