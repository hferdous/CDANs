# Quickstart

The fastest path through the library: synthetic data → fit → inspect.

```python
from cdans import CDANs
from cdans.utils import generate_synthetic_cdans

# 1. Generate a dataset with a known ground-truth structure.
dataset = generate_synthetic_cdans(
    n_vars=5,
    n_samples=500,
    tau_max=2,
    n_changing=2,
    seed=42,
)
print(f"Truth: {len(dataset.lagged_edges)} lagged edges, "
      f"{len(dataset.contemporaneous_edges)} contemp edges, "
      f"changing modules: {sorted(dataset.changing_modules)}")

# 2. Fit the model.
model = CDANs(
    tau_max=2,
    alpha=0.05,
    ci_test="fisherz",   # try "kci" for nonlinear data
)
result = model.fit(dataset.data)

# 3. Inspect what was found.
print(result.summary())
```

## What `fit` returns

A [`CDANsResult`](api/cdans.md) wrapping the discovered graph plus diagnostics:

| Attribute                     | What it is                                                        |
| ----------------------------- | ----------------------------------------------------------------- |
| `result.graph.lagged_edges`   | `set[tuple[int, int, int]]` of `(src, dst, lag)` directed edges    |
| `result.graph.contemp_adj`    | `(n, n)` int8 matrix; `1, 0` is directed, `1, 1` is undirected     |
| `result.graph.changing_modules` | `set[int]` of variable indices receiving `C → X_i`              |
| `result.timings`              | wall-clock per step                                                |
| `result.summary()`            | pretty-printed string report                                       |

## Switching the CI test

Two are bundled. Pick by name or pass an instance:

```python
CDANs(ci_test="fisherz")  # fast, linear-Gaussian baseline
CDANs(ci_test="kci")      # kernel-based; nonlinear; slower (~O(n³))
```

Or supply your own — anything implementing the [`CITest` protocol](api/ci_tests.md):

```python
class MyTest:
    name = "my_test"
    def pvalue(self, x, y, z=None):
        ...

CDANs(ci_test=MyTest())
```

## Per-step access

For research and debugging, the four steps are exposed individually:

```python
from cdans.steps import (
    discover_lagged_adjacencies,    # Step 1 (PCMCI)
    build_partial_graph,            # Step 2 (skeleton init)
    refine_skeleton,                # Step 3 (CI tests with lagged conditioning)
    orient_edges,                   # Step 4 (v-structures, Meek, independent change)
)

graph = discover_lagged_adjacencies(data, tau_max=2, ci_test="kci")
build_partial_graph(graph)
refine_skeleton(graph, data, ci_test="kci", surrogate=time_index)
orient_edges(graph, data=data, surrogate=time_index)
```

Useful when you want to use a different CI test per step (e.g. Fisher-Z for
the lagged stage and KCI for the changing-module stage), or when you want to
swap in your own implementation of one step.

## Next steps

* [Step-by-step algorithm walkthrough](algorithm.md) — the conceptual narrative
* [Reproducing the paper](experiments.md) — running the OT and x4_4 experiments
* [API reference](api/cdans.md) — every parameter, every return type
