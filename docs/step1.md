# Step 1 — Lagged adjacencies (PCMCI)

Implements **PCMCI** (Runge et al., *Sci. Adv.* 2019) — *not* PCMCI+. PCMCI+
extends PCMCI to discover contemporaneous edges in the same pass; CDANs
deliberately handles contemporaneous structure separately in
[Step 3](step3.md) with a different conditioning strategy that exploits the
lagged parents discovered here.

The implementation is **fully self-contained** — no `tigramite` dependency,
no `causal-learn` dependency. PC-stable selection, MCI conditioning with
proper time-shifting, and the CI test loop are all built directly on top of
NumPy and the pluggable [`CITest` protocol](api/ci_tests.md).

## Two-phase structure

### PC step (PC-stable, Colombo & Maathuis 2014)

For each target `X_j[t]`, prune candidate lagged parents by an iterative
process keyed on conditioning-set size:

```
candidates = all (variable, lag) columns
min_pval[c] = 1.0 for all c       # tracks strongest evidence of dependence

for cond_size = 0, 1, 2, ...:
    snapshot ranking = candidates sorted by min_pval[c] ascending
    to_remove = []
    for each c in ranked:
        cond_subset = top cond_size strongest *other* candidates from ranked
        p = ci_test(X_j[t], c, cond_subset)
        min_pval[c] = min(min_pval[c], p)
        if p > pc_alpha: to_remove.append(c)
    candidates -= to_remove
    if to_remove was empty: stop
```

Two specific properties make this **stable**:

* The candidate ranking is snapshotted *before* each iteration starts. Removals
  inside the iteration don't change which conditioning sets are tested for the
  remaining candidates.
* The min-p-value tracker preserves the strongest evidence of dependence
  observed across all subsets, so the ranking is consistent across iterations.

### MCI step

For each surviving candidate `X_i[t-lag] → X_j[t]`, run one final CI test at
the strict threshold `alpha` (default 0.05). The conditioning set is what makes
this step "Momentary":

* Other lagged parents of `X_j` (no time shift).
* Lagged parents of `X_i`, **time-shifted by `lag`**: a parent of `X_i` at
  offset `lag'` is at absolute time `(t-lag) - lag'`, which corresponds to
  column `(parent_var, lag + lag')` in the design matrix. Parents whose shifted
  lag exceeds `tau_max` are dropped.

The shift is the critical detail. Without it the conditioning would be
incoherent (asking about `X_i`'s parents at time `t` while we're testing an
edge about `X_i` at time `t-lag`).

## Why not PCMCI+?

PCMCI+ would subsume Steps 1 *and* 3 of CDANs into a single pass, but it does
not use the lagged-parent-only conditioning that gives CDANs its dimensionality
advantage on contemporaneous edges. The two methods make different statistical
trade-offs; CDANs deliberately keeps them separate so each step's conditioning
strategy is optimal for its own job.

## API

::: cdans.steps.discover_lagged_adjacencies
