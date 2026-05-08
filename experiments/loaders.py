"""Shared data loaders for the CDANs paper experiments.

The MATLAB reference repo stores each experiment's data as a folder
containing one CSV file per (pre-lagged) variable. The MLX driver
scripts read these CSVs in a specific order and stack them into a
``(T, n_vars)`` array. This module re-implements that loading step in
Python so the same data can be fed to the standalone library.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExperimentSpec:
    """Static description of a single paper experiment.

    Attributes
    ----------
    name:
        Short identifier (matches the folder name in the MATLAB repo).
    csv_files:
        Ordered list of CSV filenames as they appear in the original
        ``.mlx`` driver. Order matters — the resulting array's columns
        are in this order, which is what the paper reports.
    n_samples:
        Expected ``T`` (length of each CSV). Used as a sanity check.
    description:
        Free-text description for the README / log output.
    matlab_script:
        Filename of the original MATLAB driver, for attribution.
    """

    name: str
    csv_files: list[str]
    n_samples: int
    description: str
    matlab_script: str


# ---- Experiment registry -------------------------------------------------
#
# These match the .mlx files in the parent MATLAB repo. The CSV ordering
# is taken verbatim from the original driver scripts so that column k of
# the loaded array refers to the same variable as in the paper.

OT_EXPERIMENT = ExperimentSpec(
    name="OT",
    csv_files=[
        "13.csv", "23.csv", "33.csv", "43.csv", "53.csv", "63.csv",
        "73.csv", "83.csv", "93.csv", "103.csv", "113.csv", "123.csv",
    ],
    n_samples=86,
    description=(
        "12 pre-lagged OT variables over 86 time points. The main "
        "experiment from the CDANs paper."
    ),
    matlab_script="OT_CDANs.mlx",
)

X4_4_EXPERIMENT = ExperimentSpec(
    name="x4_4",
    csv_files=[
        "x13.csv", "x14.csv", "x22.csv", "x23.csv", "x24.csv",
        "x31.csv", "x34.csv", "x44.csv",
    ],
    n_samples=347,
    description="8 pre-lagged variables over 347 time points (variant of OT).",
    matlab_script="x4_4_CDANs.mlx",
)

EXPERIMENTS = {e.name: e for e in [OT_EXPERIMENT, X4_4_EXPERIMENT]}


def load_experiment(spec: ExperimentSpec, data_dir: Path | str) -> np.ndarray:
    """Load an experiment's data from ``data_dir`` into a ``(T, n_vars)`` array.

    Parameters
    ----------
    spec:
        One of the constants from :data:`EXPERIMENTS`.
    data_dir:
        Directory containing the CSV files. Typically points at a
        sub-folder of the MATLAB reference repo (for example
        ``CDANs/CDANs/OT`` for the OT experiment).

    Returns
    -------
    np.ndarray
        Stacked data, shape ``(spec.n_samples, len(spec.csv_files))``.

    Raises
    ------
    FileNotFoundError
        If any of the expected CSV files is missing.
    ValueError
        If a CSV's row count doesn't match ``spec.n_samples``.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")

    columns: list[np.ndarray] = []
    for fname in spec.csv_files:
        path = data_dir / fname
        if not path.is_file():
            raise FileNotFoundError(f"missing CSV in {data_dir}: {fname}")
        # ``loadtxt`` handles the BOM-prefixed first row that some of
        # the source CSVs ship with via the explicit utf-8-sig encoding.
        # The files are pure numeric (no header or comments).
        col = np.loadtxt(path, encoding="utf-8-sig")
        if col.ndim != 1:
            raise ValueError(f"{path} should contain a single column, got shape {col.shape}")
        if col.shape[0] != spec.n_samples:
            raise ValueError(
                f"{path}: expected {spec.n_samples} rows, got {col.shape[0]}"
            )
        columns.append(col)
    return np.column_stack(columns)
