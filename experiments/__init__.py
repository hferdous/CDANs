"""Paper-experiment reproducibility scripts.

This package is *not* part of the installed library (it lives outside
``src/cdans/``). It contains scripts that reproduce the experimental
runs from the CDANs paper using the standalone Python library, with
the same data files used by the MATLAB reference implementation.

Usage from the repo root::

    python -m experiments.ot    --data-dir ../CDANs/OT
    python -m experiments.x4_4  --data-dir ../CDANs/x4_4
"""

from experiments.loaders import (
    EXPERIMENTS,
    OT_EXPERIMENT,
    X4_4_EXPERIMENT,
    ExperimentSpec,
    load_experiment,
)
from experiments.runner import cli, print_report, run_experiment

__all__ = [
    "EXPERIMENTS",
    "ExperimentSpec",
    "OT_EXPERIMENT",
    "X4_4_EXPERIMENT",
    "cli",
    "load_experiment",
    "print_report",
    "run_experiment",
]
