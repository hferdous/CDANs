# Step 3 — Skeleton refinement and changing-module detection

The methodological heart of CDANs. Section 3.3 of the paper.

## The key idea

Standard PC-style algorithms test `X_i ⊥ X_j | S` over many candidate
conditioning sets `S` drawn from `(X_i, X_j)`'s contemporaneous neighbors.
For an `n`-variable problem, `S` can grow up to size `n - 2`, which is both
expensive and statistically poor (high-dimensional CI tests need huge sample
sizes).

CDANs replaces this with a tightly-bounded conditioning set:

> The lagged parents of `X_i` and `X_j` (from Step 1), **plus** at most
> `max_extra_conds` of their contemporaneous neighbors.

The size is `O(|lagged_parents|) + max_extra_conds` — typically much smaller
than `n` for sparse graphs — and it leverages the structural information
already discovered in Step 1.

## Skeleton refinement (the contemp pruning loop)

For each undirected pair `(i, j)` from the partial graph:

1. Test `X_i ⊥ X_j | lagged_parents(i) ∪ lagged_parents(j)`. If `p > alpha`,
   drop the edge with empty contemporaneous witness set.
2. Otherwise, try adding 1, 2, …, `max_extra_conds` contemporaneous neighbors
   as additional conditioning. The first subset that gives `p > alpha` drops
   the edge, with that subset recorded as the witness set.
3. If every test rejects independence, keep the edge.

The witness sets are stashed on the graph for [Step 4](step4.md) to use in
v-structure detection.

## Changing-module confirmation

Step 2 marked every variable as a candidate changing module. Step 3 confirms
or rejects each one with a single CI test:

```
X_i ⊥ C | lagged_parents(X_i)
```

If the test passes (high p-value, fail to reject independence), `X_i`'s
mechanism does not depend on the surrogate after we account for its own past;
unmark it.

This conditioning is the same trick as the skeleton refinement: the lagged
parents already explain a lot of `X_i`'s structure, so the surrogate has to
add information *beyond* what the past explains in order to count as a
changing-module signal. Without this conditioning, almost any time-series
variable would look like a changing module simply because its values drift
with `t`.

## Common failure mode: cascade from Step 1

The changing-module test depends on the lagged parents being correct. If
[Step 1](step1.md) misses an important lagged edge — particularly an
autoregressive `X_i[t-1] → X_i[t]` for a changing module — the conditioning
fails to remove the variable's own past variation, and the C-dependence test
can incorrectly accept independence.

Practical mitigation:

* Use a tighter `pc_alpha` in Step 1 (default 0.2; try 0.1).
* Use KCI rather than Fisher-Z for the CI test if the time-varying mechanism
  is nonlinear.
* Increase the sample size — KCI's power scales noticeably with `n`.

See the example in the [project's six-variable demo](algorithm.md#what-stays-undirected-and-why)
for a worked case where this cascade played out and how it was diagnosed.

## API

::: cdans.steps.refine_skeleton
