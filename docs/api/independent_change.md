# Independent change principle

Port of `infer_nonsta_dir.m` from the MATLAB reference, covering both
the fixed-bandwidth path and the GP-learned-bandwidth path (via the
``"gp"`` mode, which uses scikit-learn's ARD-RBF Gaussian process
regression in place of MATLAB's `gpml`/`minimize` machinery). See
[Step 4 — Orientation](../step4.md#independent-change-principle-details)
for the math, the auto-bandwidth heuristic, and the picking-between-modes
table.

::: cdans.independent_change.independent_change_score
