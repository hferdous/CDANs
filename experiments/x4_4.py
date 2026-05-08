"""Reproduce the x4_4 experiment from the CDANs paper.

Python port of ``CDANs/x4_4/x4_4_CDANs.mlx``. Run from the library root:

    python -m experiments.x4_4 --data-dir ../CDANs/x4_4
"""

from __future__ import annotations

from experiments.loaders import X4_4_EXPERIMENT
from experiments.runner import cli


if __name__ == "__main__":
    cli(X4_4_EXPERIMENT)
