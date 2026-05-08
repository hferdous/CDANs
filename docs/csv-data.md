# Using CDANs on your own CSV data

The synthetic-data examples elsewhere in these docs are convenient for
testing, but in practice you have a CSV file with your own variables.
This page walks through that workflow.

## Required CSV format

Before anything else, your CSV needs to be in the right shape:

* **Rows are time-ordered samples.** Oldest measurement at the top,
  newest at the bottom. CDANs uses row order as the time axis.
* **Columns are variables.** Each column is one variable's univariate
  time series. The column names become the variable names CDANs
  reports.
* **No missing values.** Drop or impute NaNs before fitting. CDANs
  raises an error if it sees any.
* **All columns numeric.** Encode or drop text columns.
* **No timestamp column.** If your CSV has one, drop it before
  fitting — CDANs treats row order as time. If you need a non-time
  surrogate (e.g. patient phase, experimental condition), pass it
  separately via the `surrogate=` argument to `CDANs(...)`.

## Minimal example

```python
import pandas as pd
from cdans import CDANs

# Load the CSV. Drop any non-numeric or timestamp columns first.
df = pd.read_csv("your_data.csv")

# Fit CDANs, preserving column names.
model = CDANs(tau_max=1, ci_test="kci")
result = model.fit(df.values, var_names=df.columns.tolist())

# The recovered graph is summarized using your column names.
print(result.graph.summary())
```

That's the whole core idea: `df.values` for the data, `df.columns.tolist()`
for the names. Pass both to `fit()`.

## Reading the output

`result.graph.summary()` prints something like:

```
TimeSeriesGraph(n_vars=4, tau_max=1)
  Lagged edges: 9
    systolic_bp(t-1) -> systolic_bp(t)
    systolic_bp(t-1) -> heart_rate(t)
    heart_rate(t-1) -> spo2(t)
    ...
  Contemporaneous edges: 2 directed, 0 undirected
    systolic_bp -> temperature
    heart_rate -> temperature
  Changing modules (1):
    heart_rate
```

The variable names match your CSV column headers. To extract
edges programmatically:

```python
g = result.graph

# Lagged edges as named tuples instead of integer indices
for src_idx, dst_idx, lag in g.lagged_edges:
    src = df.columns[src_idx]
    dst = df.columns[dst_idx]
    print(f"{src}[t-{lag}] -> {dst}[t]")

# Contemporaneous directed
for src_idx, dst_idx in g.directed_contemp_edges():
    print(f"{df.columns[src_idx]} -> {df.columns[dst_idx]}")

# Variables CDANs flagged as non-stationary
for idx in g.changing_modules:
    print(df.columns[idx])
```

## Practical notes

**Sample size.** KCI's power scales with `n`. For 4-6 variables and
`tau_max <= 2`, aim for at least 500-800 samples. Below 300, expect
more spurious edges and missed changing modules.

**`tau_max`.** Set this to the longest lag you reasonably expect to
matter. Setting it too high wastes runtime and can hurt CI-test power
(more conditioning variables); too low means missing real long-lag
effects. When in doubt, start with `tau_max=2` and check whether the
recovered structure stabilizes if you increase it.

**`ci_test`.** Use `"kci"` for nonlinear data (slower, ~`O(n³)` per
test) and `"fisherz"` for linear-Gaussian data (much faster but
linear-only). `"fisherz"` cannot detect non-stationarity by itself, so
the changing-modules step in CDANs degrades to a chance result with
`"fisherz"` — use `"kci"` if you care about which variables have
time-varying mechanisms.

**Standardization.** Not required by the algorithm, but
column-standardizing your data (subtract mean, divide by std) often
improves KCI's stability when columns have very different scales:

```python
df_std = (df - df.mean()) / df.std()
result = CDANs(tau_max=1, ci_test="kci").fit(df_std.values, var_names=df_std.columns.tolist())
```

**Multivariate time series with subjects/patients.** If your CSV is a
long-format table with multiple subjects (e.g. one row per
patient-timestep), don't fit CDANs on the concatenated rows directly —
the algorithm assumes a single time-ordered series. Either fit
per-subject and aggregate the results, or restructure into a single
representative series before fitting.

## Runnable example

A complete script that creates a demo CSV and runs the full workflow
above is in [`examples/from_csv.py`](https://github.com/hferdous/CDANs/blob/master/examples/from_csv.py).
Adapt it by replacing the synthetic-CSV-generation section with a path
to your own file.
