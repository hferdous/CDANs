# CDANs

**Temporal Causal Discovery from Autocorrelated and Non-Stationary Time Series Data.**

A pure-Python implementation of the CDANs algorithm from:

> Ferdous, M. H., Hasan, U., & Gani, M. O. (2023). *CDANs: Temporal Causal
> Discovery from Autocorrelated and Non-Stationary Time Series Data.*
> Proceedings of the 8th Machine Learning for Healthcare Conference, PMLR 219.
> [paper](https://proceedings.mlr.press/v219/ferdous23a.html) ·
> [arXiv](https://arxiv.org/abs/2302.03246)

## What CDANs is for

Time series that are both **autocorrelated** (today depends on yesterday) and
**non-stationary** (the dependence pattern itself changes over time). Three
limitations of previous methods that CDANs addresses:

1. **High dimensionality.** By using lagged parents as the conditioning set
   for contemporaneous CI tests, CDANs avoids conditioning on the full past.
2. **No lagged edges.** Most CD-NOD-style methods recover only contemporaneous
   structure; CDANs recovers both lagged and contemporaneous.
3. **Order dependence.** The discovered skeleton is independent of the
   variable ordering in the input.

## What you get

```python
from cdans import CDANs
from cdans.utils import generate_synthetic_cdans

dataset = generate_synthetic_cdans(n_vars=5, n_samples=500, tau_max=2, seed=42)
result = CDANs(tau_max=2, alpha=0.05, ci_test="kci").fit(dataset.data)

print(result.graph.lagged_edges)        # set of (src, dst, lag) tuples
print(result.graph.contemp_adj)          # n×n adjacency, 1=directed, 1+1=undirected
print(result.graph.changing_modules)     # variables whose mechanism varies with C
```

`fit()` runs four algorithm steps in sequence:

| Step | Module                                    | What it does                                   |
| :--: | :---------------------------------------- | :--------------------------------------------- |
| 1    | [Step 1 — Lagged adjacencies (PCMCI)](step1.md)        | MCI tests find `X_i[t-lag] -> X_j[t]`          |
| 2    | [Step 2 — Partial graph](step2.md)                     | Build lagged + fully-connected contemp + surrogate |
| 3    | [Step 3 — Skeleton refinement](step3.md)               | Prune contemp edges using lagged-parent conditioning |
| 4    | [Step 4 — Orientation](step4.md)                       | V-structures + Meek + independent-change principle |

For a single-page narrative covering the algorithm end-to-end, see [the
step-by-step walkthrough](algorithm.md).

## Honest scope

* **Implemented:** all four steps including the iterative
  independent-change-principle sink-finding for direction inference between
  changing modules. Self-contained KCI test (no `causal-learn` dependency)
  and self-contained PCMCI implementation (no `tigramite` dependency).
* **Not yet implemented:** the kernel-PCA driving-force visualization
  (`cd_non_con_fun.m` in the MATLAB reference) and the GP-learned kernel
  bandwidths used by the MATLAB code for `T <= 1000`.
* **Empirical defaults are heuristics.** The independent-change kernel
  bandwidth's `"auto"` mode uses an empirically-tuned multiplier of the
  median heuristic; see [the auto-bandwidth section](step4.md#auto-kernel-bandwidth)
  for why and how to override.
