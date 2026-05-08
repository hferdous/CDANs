"""Quickstart example: run CDANs on a synthetic dataset.

Run from the repo root with::

    python examples/quickstart.py

Expected runtime: a few seconds on a laptop.
"""

from cdans import CDANs
from cdans.utils import generate_synthetic_cdans


def main() -> None:
    # 1. Generate a synthetic dataset with known ground truth.
    dataset = generate_synthetic_cdans(
        n_vars=5,
        n_samples=500,
        tau_max=2,
        n_changing=2,
        seed=42,
    )

    print("=" * 60)
    print("Ground truth")
    print("=" * 60)
    print(f"Lagged edges:           {len(dataset.lagged_edges)}")
    print(f"Contemporaneous edges:  {len(dataset.contemporaneous_edges)}")
    print(f"Changing modules:       {sorted(dataset.changing_modules)}")
    print()

    # 2. Fit CDANs.
    model = CDANs(
        tau_max=2,
        alpha=0.05,
        ci_test="fisherz",   # try "kci" for nonlinear data (slower)
        verbose=True,
    )
    result = model.fit(dataset.data)

    # 3. Inspect the recovered graph.
    print()
    print("=" * 60)
    print("Recovered graph")
    print("=" * 60)
    print(result.summary())

    # 4. Compute simple recovery metrics on the lagged edges.
    true_lag = dataset.lagged_edges
    pred_lag = result.graph.lagged_edges
    tp = len(true_lag & pred_lag)
    fp = len(pred_lag - true_lag)
    fn = len(true_lag - pred_lag)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    print()
    print(f"Lagged edge recovery: precision={precision:.2f}, recall={recall:.2f} "
          f"(TP={tp}, FP={fp}, FN={fn})")


if __name__ == "__main__":
    main()
