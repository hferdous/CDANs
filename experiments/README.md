# Paper experiment reproductions

This folder contains scripts that re-run the experiments from the CDANs paper
using the standalone Python library, on the same data files that the MATLAB
reference implementation uses.

## Quick start

The experiments read from a directory of CSV files. The CSVs live in the
parent MATLAB repo (``CDANs/OT/``, ``CDANs/x4_4/``, etc.); pass the
appropriate path with ``--data-dir``. From this library's root:

```bash
python -m experiments.ot    --data-dir ../CDANs/OT
python -m experiments.x4_4  --data-dir ../CDANs/x4_4
```

Each script prints a structured report of the recovered graph: lagged
edges, contemporaneous edges (directed and undirected), changing
modules, and per-step timings.

## Available experiments

| Name      | Variables | Time points | MATLAB script             |
| --------- | --------- | ----------- | ------------------------- |
| `OT`      | 12        | 86          | `OT_CDANs.mlx`            |
| `x4_4`    | 8         | 347         | `x4_4_CDANs.mlx`          |

The other experiments in the parent repo (`x4_6`, `x6_4`, `x8.8`)
follow the same pattern; adding them is a five-line change in
`loaders.py` (register a new `ExperimentSpec`) plus a thin runner
script next to `ot.py`.

## Parameter mapping (MATLAB → Python)

The defaults in `runner.py` mirror the MATLAB driver settings:

| MATLAB                          | Python (`CDANs(...)`)                |
| ------------------------------- | ------------------------------------ |
| `alpha = 0.05`                  | `alpha=0.05`                         |
| `maxFanIn = 4`                  | `max_extra_conds=4`                  |
| `cond_ind_test = 'indtest_new_t'` (KCI) | `ci_test="kci"`              |
| `pars.if_GP1 = 1` (small T)     | KCI default (fixed-bandwidth kernel) |
| `pars.if_GP2 = 1`               | independent-change uses fixed `0.1`  |
| `pars.width = 0` → MATLAB sets to `0.1` | `independent_change_width=0.1` |
| `Type = 1` (phases 1+2+3)       | `use_independent_change=True`        |

## Honest caveats

- **Step 1 is not in the MATLAB scripts.** The MATLAB drivers feed
  pre-lagged data into a modified CD-NOD that uses a hard-coded
  initial graph (line 88 of `nonsta_cd_new.m`). The Python library
  runs the full pipeline, including Step 1 (PCMCI lagged-adjacency
  discovery), so it will discover its own lagged structure rather
  than the paper's hand-curated one. The `--tau-max=1` default keeps
  this minimal, but you can set it higher to let the algorithm look
  further back.

- **GP-learned bandwidths are not yet ported.** The MATLAB code uses
  `if_GP1=1` for KCI and `if_GP2=1` for the independent-change kernel
  when ``T <= 1000``. The Python KCI test uses fixed-bandwidth
  Gaussian kernels; this is the same algorithm with a different
  hyperparameter learning strategy. Results should be qualitatively
  similar but won't be bit-exact.

- **Output comparison is left to the caller.** The MATLAB experiments
  save `output.mat` with the recovered adjacency matrices; comparing
  them programmatically requires `scipy.io.loadmat`. The current
  scripts print the recovered graph in human-readable form so a
  reader can compare against the paper figures.

## Adding a new experiment

1. Open `loaders.py` and register a new `ExperimentSpec` constant
   with the experiment name, the CSV filenames in the order used by
   the original MATLAB script, and the expected time-series length.
2. Add it to the `EXPERIMENTS` dict.
3. Create a thin runner script that imports your spec and calls
   `cli(your_spec)`.

That's it — `runner.py` does everything else.
