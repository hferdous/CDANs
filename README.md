# CDANs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Temporal Causal Discovery from Autocorrelated and Non-Stationary Time Series Data.**

> [!NOTE]
> This repository previously contained a MATLAB implementation that
> accompanied the 2023 paper. That code had reproducibility issues and
> has been superseded by this Python library, which implements the same
> algorithm cleanly with full tests and documentation. The original
> MATLAB code is preserved at the
> [`v0.1-matlab-archive`](https://github.com/hferdous/CDANs/releases/tag/v0.1-matlab-archive)
> tag.

A Python implementation of the CDANs algorithm described in:

> Ferdous, M. H., Hasan, U., & Gani, M. O. (2023). *CDANs: Temporal Causal
> Discovery from Autocorrelated and Non-Stationary Time Series Data.*
> Proceedings of the 8th Machine Learning for Healthcare Conference, PMLR 219.
> [paper](https://proceedings.mlr.press/v219/ferdous23a.html) ·
> [arXiv](https://arxiv.org/abs/2302.03246)

CDANs is a constraint-based causal discovery algorithm designed for time
series that are both **autocorrelated** (today depends on yesterday) and
**non-stationary** (the dependence pattern itself changes over time). It
addresses three limitations of previous methods:

1. **High dimensionality** — by using lagged parents as the conditioning set
   for contemporaneous CI tests, CDANs avoids conditioning on the full past.
2. **No lagged edges** — most CD-NOD-style methods only recover
   contemporaneous structure; CDANs recovers both lagged and contemporaneous.
3. **Order dependence** — the discovered skeleton does not depend on the
   order in which variables are listed in the input.

## Installation

```bash
pip install cdans
```

For development:

```bash
git clone https://github.com/hferdous/CDANs.git
cd CDANs
pip install -e ".[dev]"
```

Optional extras:

| Extra   | Adds                                                      |
| ------- | --------------------------------------------------------- |
| `gp`    | `GPy` for kernel-width learning                           |
| `viz`   | `matplotlib`, `pydot` for graph drawing                   |
| `dev`   | `pytest`, `ruff`, `mypy`, `build`                         |
| `docs`  | `mkdocs` and theme                                        |
| `all`   | everything above                                          |

## Quickstart

```python
from cdans import CDANs
from cdans.utils import generate_synthetic_cdans

# 1. Generate a dataset with known ground truth.
dataset = generate_synthetic_cdans(
    n_vars=5, n_samples=500, tau_max=2, n_changing=2, seed=42,
)

# 2. Fit the model.
model = CDANs(tau_max=2, alpha=0.05, ci_test="fisherz")
result = model.fit(dataset.data)

# 3. Inspect the recovered graph.
print(result.summary())
print("Lagged edges:        ", result.graph.lagged_edges)
print("Contemp adjacency:\n", result.graph.contemp_adj)
print("Changing modules:    ", result.graph.changing_modules)
```

A runnable version is in [`examples/quickstart.py`](examples/quickstart.py).

## Plotting

Two graph visualizations are available via tigramite:

```python
from cdans import plot_process_graph, plot_time_series_graph

plot_process_graph(result.graph, save_path="process.png")        # aggregate
plot_time_series_graph(result.graph, save_path="time_series.png")  # time-unrolled
```

Requires the `viz` extra: `pip install "cdans[viz]"`. See
[`examples/plotting.py`](examples/plotting.py) for a complete worked example.

## Evaluation

When you have ground truth (e.g. from `generate_synthetic_cdans`), score
the recovered graph with standard structure-recovery metrics:

```python
from cdans import CDANs, evaluate_graph
from cdans.utils import generate_synthetic_cdans

dataset = generate_synthetic_cdans(n_vars=4, n_samples=800, tau_max=1, n_changing=1, seed=7)
result = CDANs(tau_max=1, ci_test="kci").fit(dataset.data)

metrics = evaluate_graph(result.graph, dataset)
print(metrics.summary())
print(f"SHD: {metrics.shd_total},  lagged TPR: {metrics.lagged.tpr:.2f},  FDR: {metrics.lagged.fdr:.2f}")
```

Reported separately for lagged edges, contemp skeleton, contemp directed,
and changing modules: TP / FP / FN / precision / recall (= TPR) / FDR / F1,
plus SHD per category and total.

## API

The library is built around two layers:

### High-level: `CDANs` class

```python
CDANs(
    tau_max=2,                # max lag
    alpha=0.05,               # CI test significance level
    pc_alpha=0.2,             # looser alpha for PC step inside Step 1
    ci_test="fisherz",        # or "kci", or your own CITest instance
    surrogate="time",         # or pass an array of shape (n_samples,)
    max_extra_conds=2,        # contemp neighbors added to lagged conds
    use_independent_change=True,
    independent_change_width="auto",  # or a positive float
    verbose=False,
)
```

`fit(data)` returns a `CDANsResult` with:

* `.graph` — `TimeSeriesGraph` (lagged edges, contemporaneous adjacency,
  changing modules)
* `.timings` — wall-clock per step
* `.config` — the configuration used
* `.summary()` — pretty-printed report

### Low-level: `cdans.steps` submodule

The four steps can be invoked individually, which is useful for research
and for swapping in your own implementations:

```python
from cdans.steps import (
    discover_lagged_adjacencies,   # Step 1
    build_partial_graph,           # Step 2
    refine_skeleton,               # Step 3
    orient_edges,                  # Step 4
)
```

Each step takes a `TimeSeriesGraph` (or produces one for Step 1) and mutates
it in place.

## Algorithm overview

CDANs has four steps, each mapped to a section of the paper:

| Step | Paper §  | Module                                    | What it does                                                                       |
| ---- | -------- | ----------------------------------------- | ---------------------------------------------------------------------------------- |
| 1    | §3.1     | `cdans.steps.step1_lagged`                | MCI tests find lagged adjacencies `X_i[t-lag] -> X_j[t]`                           |
| 2    | §3.2     | `cdans.steps.step2_partial_graph`         | Build the partial graph: lagged + fully-connected contemporaneous + surrogate `C`  |
| 3    | §3.3     | `cdans.steps.step3_skeleton`              | **The key step.** Prune contemporaneous edges using lagged-parent conditioning sets, and confirm changing modules |
| 4    | §3.4     | `cdans.steps.step4_orient`                | Orient contemp edges via v-structures + Meek's rules                               |

The graph is encoded as a `TimeSeriesGraph` (see `cdans.graph`).
Independence tests live in `cdans.ci_tests` and are pluggable.

## Project layout

```
cdans-library/
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── src/cdans/
│   ├── __init__.py
│   ├── algorithm.py           # Top-level CDANs class
│   ├── steps/                 # The four algorithm steps
│   ├── ci_tests/              # CI test interface + implementations
│   ├── graph/                 # TimeSeriesGraph + Meek's rules
│   └── utils/                 # Lagging helpers + synthetic data
├── tests/
│   ├── test_lagging.py
│   ├── test_synthetic.py
│   ├── test_graph.py
│   ├── test_ci_tests.py
│   └── test_end_to_end.py
└── examples/
    └── quickstart.py
```

## Status of features vs. the MATLAB reference

The reference MATLAB implementation lives in `CDANs/OT/` of the parent repo.
Status of the Python port:

| Feature                                                   | Status |
| --------------------------------------------------------- | ------ |
| Step 1: MCI-based lagged adjacencies                      | ✓      |
| Step 2: partial graph construction                        | ✓      |
| Step 3: optimized CI tests (lagged-parent conditioning)   | ✓      |
| Step 3: changing-module detection                         | ✓      |
| Step 4: surrogate orientation `C -> X`                    | ✓      |
| Step 4: v-structure orientation                           | ✓      |
| Step 4: Meek's rules R1–R4                                | ✓      |
| Synthetic data generator                                  | ✓      |
| Independent change principle for direction inference      | ✓ (`infer_nonsta_dir.m` non-GP path; plus a GP-bandwidth path via scikit-learn for the small-T regime) |
| Driving-force visualization (kPCA)                        | ☐ (TODO; the MATLAB `cd_non_con_fun.m` is not yet ported)  |
| MIMIC-III data preparation                                | Out of scope (clinical data access required) |

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

## Citing

If you use this software, please cite the paper:

```bibtex
@inproceedings{ferdous2023cdans,
  title     = {CDANs: Temporal Causal Discovery from Autocorrelated and Non-Stationary Time Series Data},
  author    = {Ferdous, Muhammad Hasan and Hasan, Uzma and Gani, Md Osman},
  booktitle = {Proceedings of the 8th Machine Learning for Healthcare Conference},
  series    = {Proceedings of Machine Learning Research},
  volume    = {219},
  year      = {2023},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v219/ferdous23a.html},
}
```

## Acknowledgements

CDANs builds on prior work in constraint-based causal discovery, notably:

* **CD-NOD** (Huang & Zhang, 2017) — provides the surrogate-variable
  construction and the changing-module formalism. The Python implementation
  in [`causal-learn`](https://github.com/py-why/causal-learn) was used as
  a reference when porting the KCI test that ships with this library.
* **PCMCI** (Runge et al., 2019) — provides the MCI conditional
  independence framework that Step 1 implements. The original reference
  implementation is in [`tigramite`](https://github.com/jakobrunge/tigramite);
  CDANs ships an independent self-contained Python port.
* **KCI** (Zhang, Peters, Janzing & Schölkopf, 2011) — kernel-based
  conditional independence test. CDANs includes an in-house implementation
  in `cdans.ci_tests.kci`, validated to produce identical p-values to
  the `causal-learn` reference on a battery of synthetic test cases.
* **Meek's rules** (Meek, 1995) — used in Step 4 for orientation propagation.

## Documentation

A full documentation site is included. To build and serve it locally:

```bash
pip install "cdans[docs]"
mkdocs serve
```

The site has installation, quickstart, a step-by-step algorithm walkthrough,
per-step deep dives, an auto-generated API reference, and a guide to
reproducing the paper experiments.

## License

MIT — see [`LICENSE`](LICENSE).
