# Evaluation

When you have ground truth — typically because you generated the data
yourself with [`generate_synthetic_cdans`](api/utils.md) — you can score
the recovered graph with standard structure-recovery metrics.

## Quick example

```python
from cdans import CDANs, evaluate_graph
from cdans.utils import generate_synthetic_cdans

dataset = generate_synthetic_cdans(
    n_vars=5, n_samples=400, tau_max=2, n_changing=2, seed=42,
)
result = CDANs(tau_max=2, ci_test="kci").fit(dataset.data)

metrics = evaluate_graph(result.graph, dataset)
print(metrics.summary())
```

Output:

```
Structure recovery metrics
============================================================
  Lagged edges               TP= 11 FP= 12 FN=  2  P=0.48 R=0.85 F1=0.61 FDR=0.52
  Contemp skeleton           TP=  0 FP=  3 FN=  2  P=0.00 R=0.00 F1=0.00 FDR=1.00
  Contemp directed           TP=  0 FP=  3 FN=  2  P=0.00 R=0.00 F1=0.00 FDR=1.00
  Changing modules           TP=  2 FP=  2 FN=  0  P=0.50 R=1.00 F1=0.67 FDR=0.50

  SHD (lagged):    14
  SHD (contemp):   5
  SHD (total):     19
```

## What's reported

The metrics are computed separately for four edge categories:

| Category               | Compared as                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| **Lagged edges**       | Strict ``(src, dst, lag)`` match — lagged edges are always directed.    |
| **Contemp skeleton**   | Adjacency only — undirected pairs ``(i, j)``. Direction ignored. |
| **Contemp directed**   | Strict directed-edge match. Undirected predicted edges count as neither TP nor FP for a directed truth edge. |
| **Changing modules**   | Set comparison on variable indices receiving ``C → X_i``.        |

For each category, a [`GraphMetrics`](api/evaluation.md#cdans.evaluation.GraphMetrics)
object exposes:

| Attribute    | Definition                              |
| ------------ | --------------------------------------- |
| `tp`         | True positives (predicted ∩ truth)      |
| `fp`         | False positives (predicted − truth)     |
| `fn`         | False negatives (truth − predicted)     |
| `precision`  | `TP / (TP + FP)`                        |
| `recall`     | `TP / (TP + FN)`                        |
| `tpr`        | True positive rate (alias for recall)   |
| `fdr`        | False discovery rate `= 1 − precision`  |
| `f1`         | Harmonic mean of precision and recall   |

## Structural Hamming Distance (SHD)

Two SHD numbers are reported:

* **`shd_lagged`** — symmetric difference of the lagged-edge sets.
  Each missing or extra edge counts as 1 unit.
* **`shd_contemp`** — PDAG-aware. For each unordered pair `(i, j)`, the
  state is one of `{no edge, i→j, j→i, undirected}`. One SHD unit is
  added per state mismatch. A reversed direction or an
  undirected-vs-directed disagreement is one unit; a missing edge is
  also one unit.

`shd_total = shd_lagged + shd_contemp`. Changing-module disagreements are
*not* folded into SHD — they're reported separately, since they're a
binary attribute per variable rather than an edge.

## Without a `SyntheticDataset`

If you have ground truth from elsewhere (e.g. a real dataset with known
structure), build a `TimeSeriesGraph` directly:

```python
from cdans import TimeSeriesGraph, evaluate_graph

truth = TimeSeriesGraph(n_vars=5, tau_max=2)
truth.add_lagged_edge(0, 1, lag=1)
truth.add_lagged_edge(2, 3, lag=2)
truth.orient_contemp(0, 4)
truth.mark_changing(2)

metrics = evaluate_graph(predicted_graph, truth)
```

## Aggregate (full) TPR / FDR

For papers and benchmarking, you usually want a single overall TPR /
FDR / F1 number per fit instead of one per category. Two aggregates
are computed automatically:

```python
metrics.total.tpr           # overall true-positive rate
metrics.total.fdr           # overall false-discovery rate
metrics.total.f1            # overall F1
metrics.total.precision     # overall precision
```

`total` pools the TP / FP / FN counts across **lagged edges**,
**contemporaneous directed edges**, and **changing modules** — each
edge or marked module counts once. This is the standard "full TPR /
FDR" causal-discovery papers report.

A more lenient skeleton-level companion is also available:

```python
metrics.total_skeleton.tpr  # treats wrong-direction contemp edges as TPs
```

`total_skeleton` uses the contemp **adjacency** instead of directed
edges. The difference between `total` and `total_skeleton` tells you
how much of the algorithm's error comes from direction mistakes versus
finding the wrong adjacencies.

## Just the SHD

If you only want a single scalar:

```python
from cdans import shd
total = shd(predicted_graph, dataset)
```

For the full API reference (signatures, defaults, every attribute), see
[the evaluation API page](api/evaluation.md).
