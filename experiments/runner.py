"""Common runner for the paper experiments.

Mirrors the parameter choices made by the MATLAB ``.mlx`` drivers so
the Python output is comparable to what's in the paper:

* ``alpha = 0.05`` — significance level for CI tests.
* ``ci_test = "kci"`` — kernel-based CI test for both Step 1 and Step 3
  (the MATLAB scripts pick GP-learned KCI for ``T <= 1000``; we use the
  fixed-bandwidth KCI ported here, which is close in spirit).
* ``max_extra_conds = 4`` — corresponds to MATLAB's ``maxFanIn = 4``.
* ``independent_change_width = 0.1`` — matches the MATLAB
  ``pars.width = 0`` fallback (which becomes ``0.1`` inside
  ``infer_nonsta_dir.m``).
* ``use_independent_change = True`` — MATLAB ``Type = 1``.

Pass ``ci_test="fisherz"`` for a fast linear-Gaussian baseline that's
useful when iterating on parameters.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cdans import CDANs, CDANsResult
from experiments.loaders import EXPERIMENTS, ExperimentSpec, load_experiment


def run_experiment(
    spec: ExperimentSpec,
    data_dir: Path | str,
    *,
    tau_max: int = 1,
    alpha: float = 0.05,
    pc_alpha: float = 0.2,
    ci_test: str = "kci",
    max_extra_conds: int = 4,
    independent_change_width: float | str = 0.1,
    verbose: bool = False,
) -> tuple[np.ndarray, CDANsResult]:
    """Load ``spec``'s data from ``data_dir`` and run the standalone CDANs library.

    Returns ``(data, result)``.
    """
    data = load_experiment(spec, data_dir)
    if verbose:
        print(f"[{spec.name}] loaded data: shape {data.shape}")

    model = CDANs(
        tau_max=tau_max,
        alpha=alpha,
        pc_alpha=pc_alpha,
        ci_test=ci_test,
        max_extra_conds=max_extra_conds,
        use_independent_change=True,
        independent_change_width=independent_change_width,
        verbose=verbose,
    )
    t0 = time.perf_counter()
    result = model.fit(data)
    if verbose:
        print(f"[{spec.name}] fit completed in {time.perf_counter() - t0:.1f}s")
    return data, result


def print_report(spec: ExperimentSpec, data: np.ndarray, result: CDANsResult) -> None:
    """Print a human-readable summary."""
    print("=" * 70)
    print(f"Experiment: {spec.name}  (data shape {data.shape})")
    print(f"Source:    {spec.description}")
    print(f"MATLAB:    {spec.matlab_script}")
    print("=" * 70)
    print(result.summary())


def cli(spec: ExperimentSpec) -> None:
    """A minimal CLI wrapper used by per-experiment scripts."""
    parser = argparse.ArgumentParser(description=f"Run the {spec.name} experiment.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=f"Directory containing the {spec.name} CSV files (e.g. parent_repo/CDANs/{spec.name}).",
    )
    parser.add_argument(
        "--ci-test", default="kci", choices=["fisherz", "kci"]
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--pc-alpha", type=float, default=0.2)
    parser.add_argument("--tau-max", type=int, default=1)
    parser.add_argument("--max-extra-conds", type=int, default=4)
    parser.add_argument(
        "--independent-change-width",
        default="0.1",
        help='Either a float, or the string "auto" to use median-heuristic.',
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Allow --independent-change-width=auto on the CLI.
    icw: float | str
    if args.independent_change_width == "auto":
        icw = "auto"
    else:
        icw = float(args.independent_change_width)

    data, result = run_experiment(
        spec,
        data_dir=args.data_dir,
        tau_max=args.tau_max,
        alpha=args.alpha,
        pc_alpha=args.pc_alpha,
        ci_test=args.ci_test,
        max_extra_conds=args.max_extra_conds,
        independent_change_width=icw,
        verbose=args.verbose,
    )
    print_report(spec, data, result)


__all__ = ["EXPERIMENTS", "ExperimentSpec", "cli", "print_report", "run_experiment"]
