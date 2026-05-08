# Reproducing the paper experiments

The `experiments/` folder in the source repo contains scripts that re-run
the experiments from the CDANs paper using this library, on the same data
files used by the MATLAB reference implementation.

## Quick run

From the repo root, with the parent MATLAB repo's data folder available:

```bash
python -m experiments.ot   --data-dir ../CDANs/OT
python -m experiments.x4_4 --data-dir ../CDANs/x4_4
```

Each script prints a structured report: lagged edges, contemporaneous
edges (directed and undirected), changing modules, and per-step timings.

## Available experiments

| Name      | Variables | Time points | MATLAB script             |
| --------- | --------- | ----------- | ------------------------- |
| `OT`      | 12        | 86          | `OT_CDANs.mlx`            |
| `x4_4`    | 8         | 347         | `x4_4_CDANs.mlx`          |

The other experiments in the parent repo (`x4_6`, `x6_4`, `x8.8`) follow the
same pattern; adding them is a five-line change in `experiments/loaders.py`
(register a new `ExperimentSpec`) plus a thin runner script.

## Parameter mapping (MATLAB → Python)

The defaults in `experiments/runner.py` mirror the MATLAB driver settings:

| MATLAB                                | Python (`CDANs(...)`)                  |
| ------------------------------------- | -------------------------------------- |
| `alpha = 0.05`                        | `alpha=0.05`                           |
| `maxFanIn = 4`                        | `max_extra_conds=4`                    |
| `cond_ind_test = 'indtest_new_t'`     | `ci_test="kci"`                        |
| `pars.if_GP1 = 1` (small T)           | KCI default (fixed-bandwidth kernel)   |
| `pars.if_GP2 = 1`                     | independent-change uses fixed `0.1`    |
| `pars.width = 0` → MATLAB sets `0.1`  | `independent_change_width=0.1`         |
| `Type = 1` (phases 1+2+3)             | `use_independent_change=True`          |

## Caveats

* **Step 1 isn't in the MATLAB scripts.** The MATLAB drivers feed pre-lagged
  data into a modified CD-NOD that uses a hard-coded initial graph
  (`nonsta_cd_new.m` line 88). The Python library runs the full pipeline,
  including Step 1 (PCMCI lagged-adjacency discovery), so it discovers its
  own lagged structure rather than the paper's hand-curated initial graph.
  Use `--tau-max 1` to keep this minimal, or higher values to look further
  back.
* **GP-learned bandwidths: partially ported.** The MATLAB code uses
  `if_GP1=1` for KCI and `if_GP2=1` for the independent-change kernel
  when `T <= 1000`. The independent-change side is now ported via
  `independent_change_width="gp"`, which fits an ARD-RBF Gaussian
  process via scikit-learn and uses the marginal-likelihood-optimized
  length scales as bandwidths. The KCI test still uses fixed-bandwidth
  Gaussian kernels — that's the remaining gap. Results should be
  qualitatively similar to the MATLAB reference but won't be bit-exact.
* **Output comparison is left to the caller.** The MATLAB experiments
  save `output.mat` with the recovered adjacency matrices; comparing
  them programmatically would require `scipy.io.loadmat`.

## Adding a new experiment

1. Open `experiments/loaders.py` and register a new `ExperimentSpec`
   constant with the experiment name, the CSV filenames in the order used
   by the original MATLAB script, and the expected time-series length.
2. Add it to the `EXPERIMENTS` dict in the same file.
3. Create a thin runner script next to `ot.py` that imports your spec
   and calls `cli(your_spec)`.

That's it — `runner.py` does everything else.
