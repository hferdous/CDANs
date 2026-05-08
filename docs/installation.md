# Installation

## From PyPI (recommended)

```bash
pip install cdans
```

This pulls in the runtime dependencies — NumPy, SciPy, pandas, NetworkX,
scikit-learn, tqdm. No causal-discovery libraries (no `tigramite`, no
`causal-learn`) are required at runtime; PCMCI and KCI are both
implemented in-house.

## From source

```bash
git clone https://github.com/hferdous/CDANs.git
cd CDANs
pip install -e .
```

Use `-e .` for an editable install if you plan to modify the library;
omit `-e` for a regular install.

## Optional extras

| Extra   | Adds                                                |
| ------- | --------------------------------------------------- |
| `gp`    | GPy for future GP-based kernel-width learning       |
| `viz`   | matplotlib + pydot for graph drawing                |
| `dev`   | pytest, ruff, mypy, build (for development)         |
| `docs`  | mkdocs + mkdocs-material + mkdocstrings (this site) |
| `all`   | everything above                                    |

```bash
pip install "cdans[dev]"        # for development
pip install "cdans[docs]"       # to build this docs site
pip install "cdans[all]"        # everything
```

## Verify the install

```bash
python -c "import cdans; print(cdans.__version__)"
```

## Run the test suite

```bash
pip install "cdans[dev]"
pytest
```

Around 90 tests, finishes in a few seconds. They cover the algorithm steps
end-to-end on synthetic data, the KCI port (validated against the
`causal-learn` reference on a battery of test cases), and the MATLAB-faithful
direction inference.
