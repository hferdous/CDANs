# Step 4 — Orientation

After [Step 3](step3.md) produces the contemporaneous skeleton and the
confirmed changing-module set, Step 4 determines as many edge directions
as the algorithm can. Four sub-passes, run in this order:

## 1. Surrogate orientation

Every variable still in `changing_modules` is treated as receiving an
implicit directed edge `C → X_i`. This is encoded by membership in the
set rather than as a materialized adjacency-matrix entry — the surrogate
is special and has no contemporaneous edges to it.

## 2. V-structure detection

For every pair of contemporaneous neighbors `(a, c)` of a middle node `b`:

* If `a` and `c` are adjacent → shielded triple; skip.
* If `b` is in the witness set that separated `a` and `c` in Step 3 →
  `b` is a fork or chain on the `a-c` path; skip.
* Otherwise → orient as `a → b ← c` (unshielded collider).

A small CDANs-specific tie-breaker: if exactly one of `a` and `c` is a
changing module while `b` is not, prefer the changing module as cause when
only one of the two edges can be oriented. Consistent with the
surrogate-priority intuition from CD-NOD.

## 3. Independent-change-principle sink-finding

The novel orientation primitive. Used for undirected edges between two
changing modules — exactly the case where v-structures and Meek's rules
together cannot decide direction.

### Algorithm

```
candidates = {changing modules with at least one undirected edge}

while len(candidates) > 1:
    for each candidate c in candidates:
        parents_of_c = {everyone with an edge into c}
                     ∪ {everyone with an undirected edge to c}
        score[c] = independent_change_score(
            parents=data[:, parents_of_c],
            effect =data[:, c],
            surrogate=surrogate,
        )

    sink = argmin(scores)
    orient every undirected edge from sink's parents toward sink
    candidates.remove(sink)
```

The intuition: under the independent-change principle (Huang & Zhang, 2017),
the *causal mechanism* `P(effect | parents)` and the *cause distribution*
`P(parents)` should change **independently** as the system evolves under
non-stationarity. The score quantifies the dependence between kernel
representations of these two distributions as functions of the surrogate.
Lower means more independent → stronger evidence the assumed direction is
correct → the candidate really is a sink.

### Independent change principle: details

For each candidate-as-sink, the score is computed as follows.

1. **Standardize** `parents`, `effect`, and `surrogate` (zero mean, unit std).
2. Build Gaussian kernel matrices `K_x`, `K_y`, `K_t` for the three.
3. **Module 1 — `P(effect | parents)`:**
   `Mₗ = (1/T²) · K_t · (K_x³ ⊙ (invK · K_y · invK)) · K_t`
   where `invK = (K_x ⊙ K_t + λ I)⁻¹`. Convert `Mₗ` to a Gaussian-kernel
   form via the inner-product → squared-distance identity.
4. **Module 2 — `P(parents)`:**
   `Mₗ' = K_t · (K_t + λ I)⁻¹ · K_x · (K_t + λ I)⁻¹ · K_t`. Convert similarly.
5. **HSIC-style dependence statistic:** double-center both modules with
   `H = I - 1/T · 1 1ᵀ`, then compute `(1/T²) · sum(M_centered ⊙ M'_centered)`.

This is a direct port of `infer_nonsta_dir.m` from the MATLAB reference.

### Auto kernel bandwidth

The X/Y kernel bandwidth matters a lot for this score. The default
`independent_change_width="auto"` selects via:

> bandwidth = `0.05 · median_bandwidth(standardized parents)`

The 0.05 multiplier is **empirically tuned**: the algorithm needs a *narrow*
kernel (the MATLAB reference hard-codes `width = 0.1` on standardized data),
but a fixed constant doesn't adapt to non-Gaussian data. A bandwidth sweep
on synthetic non-stationary `X → Y` data over 60 trials:

| fraction of median | direction-recovery accuracy |
| -----------------: | --------------------------: |
| 0.02–0.08          | ~92–93%                     |
| 0.10               | 72%                         |
| 0.15               | 30%                         |
| 1.00 (pure median) | ~30%                        |

The algorithm has a sharp performance cliff above fraction ≈ 0.1, so 0.05
is a safe choice in the working range.

### GP-learned kernel bandwidth

Setting `independent_change_width="gp"` triggers an alternative bandwidth
selection: fit a Gaussian process with an ARD-RBF kernel on
`(parents, surrogate) → effect` and use the marginal-likelihood-optimized
length scales as the bandwidths.

```python
CDANs(independent_change_width="gp")
```

This is the Python analog of the MATLAB reference's `if_GP2=1` path. It's
adaptive (the bandwidths change per call to fit the local data) but
costs roughly 10× more per score call than `"auto"`.

**Empirical comparison on a 6-variable lag-3 DGP** with two changing
modules connected by a contemporaneous edge that's hard for IC to orient:

| Bandwidth mode | Direction recovery (8 seeds) | Per-call time |
| --------------- | ----------------------------: | -------------: |
| `"auto"`        | 2/8 (25%)                     | ~3 s          |
| `"gp"`          | 5/8 (62%)                     | ~30 s         |

GP isn't a strict improvement on every seed; it can overfit at small `T`
or pick suboptimal length scales when the optimizer hits a local minimum.
But on hard cases where `"auto"` fails consistently across all manual
bandwidths, `"gp"` is often the only thing that flips them.

### Picking between modes

| Use ... | when |
| ------- | ---- |
| `"auto"` (default) | Starting from scratch, throughput matters, or the recovered graph already looks right. Run this first. |
| `"gp"` | Skeleton looks correct but a contemp edge between two changing modules is oriented wrong, AND you've checked across multiple seeds. Also: if you're trying to reproduce MATLAB-paper numbers for `T ≤ 1000`. |
| Manual float | Reproducing a specific reference configuration (`0.1` for the MATLAB original) or you've done your own bandwidth tuning on held-out data. |

**When `"auto"` and `"gp"` disagree on a direction:** trust `"gp"` more on
small `T` (≤ 1000); on larger `T` neither bandwidth heuristic is
necessarily correct on its own — run multiple seeds and report the
majority-vote orientation, or fall back to whatever orientation evidence
is available (v-structures, prior knowledge, downstream task accuracy).

To override: pass a positive float for fully manual control:

```python
CDANs(independent_change_width=0.1)    # MATLAB-faithful fixed value
CDANs(independent_change_width=0.03)   # a bit narrower
```

The surrogate-side bandwidth uses the standard median heuristic on the
*standardized* surrogate, so users can pass raw time indices without scaling.

## 4. Meek's rules

Apply rules R1–R4 repeatedly until nothing changes.

* **R1** — `a → b` and `b — c` and `a` not adjacent to `c` ⇒ `b → c`.
  (No new v-structure.)
* **R2** — `a → b → c` and `a — c` ⇒ `a → c`. (No cycle.)
* **R3** — `a — b`, `a — c`, `a — d`, `c → b`, `d → b`, `c` and `d` not
  adjacent ⇒ `a → b`.
* **R4** — `a — b`, `a — c`, `c → d`, `d → b`, `a` and `d` not adjacent
  ⇒ `a → b`.

The rules run on the contemporaneous PDAG only — lagged edges and surrogate
edges are excluded.

## API

::: cdans.steps.orient_edges

::: cdans.independent_change.independent_change_score
