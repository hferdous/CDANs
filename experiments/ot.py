"""Reproduce the OT experiment from the CDANs paper.

This is a Python port of the MATLAB driver at ``CDANs/OT/OT_CDANs.mlx`` in the
parent repo. Run from the cdans-library root, pointing ``--data-dir`` at the
folder containing the CSV files (typically ``../CDANs/OT``):

    python -m experiments.ot --data-dir ../CDANs/OT

Optional flags::

    --ci-test fisherz                 # fast Fisher-Z baseline (vs. KCI default)
    --alpha 0.01                      # tighter significance
    --independent-change-width auto   # median-heuristic kernel bandwidth
    --verbose                         # per-step progress
"""

from __future__ import annotations

from experiments.loaders import OT_EXPERIMENT
from experiments.runner import cli


if __name__ == "__main__":
    cli(OT_EXPERIMENT)
