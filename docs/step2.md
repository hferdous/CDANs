# Step 2 — Build the partial graph

A bookkeeping step. Given the lagged structure from [Step 1](step1.md):

1. Add an undirected contemporaneous edge between every pair `(i, j)` of
   variables. This is the fully-connected starting skeleton; [Step 3](step3.md)
   will thin it.
2. Tentatively mark every variable as a changing module. [Step 3](step3.md)
   will prune via CI tests against the surrogate.

No statistical tests are run here. Lagged edges are untouched.

## Why this is its own step

Two reasons:

1. **Separation of concerns.** Steps 1 and 3 are statistical tests; Step 2 is
   pure data-structure manipulation. Keeping it separate makes the boundary
   between "what we observed" and "what we initialize" explicit.
2. **Easier to swap in different starting structures.** A user with prior
   knowledge can replace [`build_partial_graph`](api/steps.md) with their
   own initializer (e.g., one that excludes physically impossible
   contemporaneous edges) without touching Step 3 or Step 4.

## API

::: cdans.steps.build_partial_graph
