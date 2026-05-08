# How CDANs works, step by step

Given a multivariate time series `X` of shape `(T, n)` — T time points, n variables —
CDANs returns a `TimeSeriesGraph` containing lagged edges, contemporaneous edges
(oriented where possible), and the set of variables whose generating mechanism
changes over time.

## Setup

Before any step runs, two things happen.

**Lagged design matrix.** For maximum lag `τ_max`, the time series is split into
a target matrix `Y` of shape `(T-τ_max, n)` containing values at time
`t = τ_max, ..., T-1`, and a lagged regressor matrix `X_lagged` of shape
`(T-τ_max, n·τ_max)` whose columns hold past values at offsets 1 through `τ_max`.
Each column has a `(variable, lag)` label, so column `c` represents `X_var[t-lag]`.
This is what lets every CI test in Steps 1 and 3 pull out a lagged variable by
indexing.

**Surrogate variable C.** The distribution-shift surrogate is by default the
time index `[0, 1, ..., T-1]`, but you can pass any `(T,)` array — domain index
for heterogeneous data, regime label, anything that captures non-stationarity.

## Step 1 — Lagged adjacency discovery (PCMCI)

The goal is to find every edge of the form `X_i[t-lag] → X_j[t]`. Two passes per
target variable.

**PC step.** For each target `X_j[t]`, every column of `X_lagged` starts as a
candidate parent. We iterate the conditioning-set size `cond_size` from 0 upward.
At the start of each iteration we *snapshot* the candidate ranking — strongest-first
by smallest p-value seen so far — and then for each candidate `c`, we test
`X_c ⊥ X_j[t] | top-cond_size strongest others`. If the p-value exceeds `pc_alpha`
(loose threshold, default 0.2) we mark `c` for removal. Removals happen at the
end of the iteration so the ordering doesn't drift mid-pass — that's the
"stable" in PC-stable. We stop when no candidate falls in a round, or when
`cond_size` would exceed the number of survivors.

**MCI step.** For each surviving candidate `X_i[t-lag]` of target `X_j`, we run
one final test at the strict threshold `alpha` (default 0.05). The conditioning
set has two parts:

- **Other lagged parents of X_j**, unshifted — they're already in the right
  time frame.
- **Lagged parents of X_i**, time-shifted by `lag`. A parent of `X_i` at offset
  `lag'` is at absolute time `(t-lag) - lag'`, so it appears in our design
  matrix at column `(parent_var, lag + lag')`. Parents whose shifted lag exceeds
  `τ_max` get dropped — we can't represent them.

The shift is what makes MCI "momentary": we condition on the source's own past
as it was *at the time the candidate edge would have fired*, not its current
past.

Surviving candidates become entries in `graph.lagged_edges`.

## Step 2 — Build the partial graph

This is bookkeeping, not statistics.

- Add an undirected edge between every pair `(i, j)` of contemporaneous variables
  (fully-connected skeleton, to be thinned in Step 3).
- Tentatively mark *every* variable as a changing module (Step 3 will prune via
  CI tests against the surrogate).

The lagged edges from Step 1 stay untouched.

## Step 3 — Refine the contemporaneous skeleton and confirm changing modules

This is the methodological heart of CDANs. It uses the lagged structure from
Step 1 to make contemporaneous testing tractable.

**Skeleton refinement.** For each undirected pair `(i, j)`, the conditioning set
isn't all the other contemporaneous variables (that would be standard PC's
expensive choice). Instead we use the **union of the lagged parents of X_i and
X_j**, optionally augmented by up to `max_extra_conds` of their contemporaneous
neighbors.

The procedure:

1. Test `X_i ⊥ X_j | lagged_parents(i) ∪ lagged_parents(j)`. If `p > alpha`,
   drop the edge with empty contemporaneous witness.
2. If kept, try adding 1, 2, ..., `max_extra_conds` contemporaneous neighbors
   as additional conditioning. The first subset that gives `p > alpha` drops
   the edge, recording those contemporaneous vars as the witness set.
3. Edges that survive every test stay.

The witness sets are stashed on the graph for Step 4 to use in v-structure
detection.

The conditioning-set size is bounded by `|lagged parents| + max_extra_conds` —
typically much smaller than the full neighbor set PC would condition on, which
is what gives CDANs its dimensionality advantage on n-variable problems with
≪n true parents per variable.

**Changing-module confirmation.** Every variable was tentatively marked as
changing in Step 2. Now for each one we test `X_i ⊥ C | lagged_parents(i)`.
If the test passes (p-value high → fail to reject independence), `i` is
*unmarked* — its mechanism doesn't depend on the surrogate, so no `C → X_i`
edge.

This conditioning is the same trick: the lagged parents already explain a lot
of `X_i`'s variation, so the surrogate has to add information *beyond* what
the past explains in order to register as a changing-module edge.

## Step 4 — Orient contemporaneous edges

Four sub-passes.

**Surrogate orientation.** Every variable still in `changing_modules` gets a
directed edge `C → X_i`, encoded by membership in the set (the surrogate is
treated specially; no contemporaneous edge to it is materialized).

**V-structure detection.** Walk every middle node `b` and look at every
unordered pair of its contemporaneous neighbors `(a, c)`. Three possibilities:

- `a` and `c` are adjacent → shielded triple, skip.
- A witness from Step 3 separated `a` and `c`, and `b` was *in* that witness →
  `b` is a fork or chain on the `a-c` path, not a collider, skip.
- Otherwise → orient as `a → b ← c` (an unshielded collider).

A small CDANs-specific tie-breaker: if exactly one of `(a, c)` is a changing
module while `b` is not, the changing module is preferred as cause when only
one of the two edges can be oriented. This is consistent with the
surrogate-priority intuition from CD-NOD.

**Independent-change-principle sink-finding.** For undirected edges between two
changing modules, run an iterative algorithm:

1. Candidate pool = changing modules with at least one undirected edge.
2. For each candidate, treat its current parents (everyone with an edge
   pointing into it, plus all undirected neighbors) as a joint cause set
   and score how independently `P(parents)` and `P(candidate | parents)`
   change with the surrogate.
3. Pick the candidate with the **lowest** score — that's the most-confident
   sink — and orient all its undirected edges inward.
4. Remove from the pool, repeat until ≤ 1 candidate remains.

The score is a kernel-derived dependence statistic; lower means the modules
change more independently, which is the signature of a correct causal direction.
See [Step 4 — Orientation](step4.md#independent-change-principle-details) for
the math and the auto-bandwidth heuristic.

**Meek's rules.** Apply the four standard orientation propagation rules
(R1–R4) repeatedly until nothing changes. R1 forbids new v-structures, R2
forbids cycles, R3 and R4 propagate orientations through specific configurations
of three or four nodes. Lagged edges and surrogate edges are excluded from rule
application — the rules run on the contemporaneous PDAG only.

## What you get back

A `CDANsResult` with:

- `graph.lagged_edges` — set of `(source, target, lag)` tuples, all directed
  `past → present`.
- `graph.contemp_adj` — `(n, n)` int8 matrix; `1, 0` is directed, `1, 1` is
  undirected, `0, 0` is no edge.
- `graph.changing_modules` — set of variable indices receiving `C → X_i`.
- `timings` — wall-clock per step.

## What stays undirected and why

After all four sub-passes, an undirected edge can remain in the output for two
reasons:

1. **Genuinely undecidable from observational data.** Two variables in the
   same Markov equivalence class with no v-structure constraint and no
   changing-module evidence. This is fundamental, not an algorithm limitation.
2. **The independent-change pool was too small.** The sink-finding loop needs
   at least two candidates (changing modules with undirected edges) to make
   progress. If only one such candidate exists, the loop exits and any
   undirected edges incident to it remain.

In practice the second case is more common on real data, and it's a sign
worth heeding: if an edge you expected to be directed comes back undirected,
the algorithm is telling you it doesn't have enough evidence.
