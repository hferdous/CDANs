# Changelog

All notable changes to this project are documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased

First public release. Replaces the MATLAB reference implementation with
a standalone Python library.

### Added

- **Algorithm.** All four steps of CDANs in pure Python:
  - Step 1: PCMCI for lagged-adjacency discovery (PC-stable + MCI with
    proper time-shifting). Self-contained, no `tigramite` runtime
    dependency.
  - Step 2: Partial-graph initialization.
  - Step 3: Contemporaneous skeleton refinement using lagged-parent
    conditioning, plus changing-module confirmation.
  - Step 4: Orientation via v-structure detection, the iterative
    independent-change-principle sink-finding, and Meek's rules.
- **Conditional independence tests.** Two bundled implementations
  with a pluggable `CITest` protocol:
  - `FisherZ` for linear-Gaussian baselines.
  - `KCITest` — kernel-based, self-contained (no `causal-learn`
    dependency), validated bit-for-bit identical p-values to the
    `causal-learn` reference on a battery of test cases.
- **Independent-change-principle direction inference.** Direct port of
  the non-GP path of MATLAB's `infer_nonsta_dir.m`. Three bandwidth-
  selection modes:
  - `"auto"` — empirically-tuned median heuristic (fast).
  - `"gp"` — Gaussian-process marginal-likelihood-optimized length
    scales via scikit-learn (adaptive, slower).
  - manual float for paper reproduction (`0.1` matches the MATLAB
    reference).
- **Synthetic data generator** with known ground truth
  (`generate_synthetic_cdans`).
- **Evaluation metrics:** TPR, FDR, FPR, F1, precision, recall plus
  PDAG-aware SHD, computed per edge category (lagged, contemp skeleton,
  contemp directed, changing modules) and as aggregate "full TPR/FDR"
  numbers.
- **Plotting** via tigramite (optional `[viz]` extra):
  - `plot_process_graph` — aggregate process view.
  - `plot_time_series_graph` — time-unrolled view, surrogate `C`
    drawn as a top row.
- **Paper-experiment reproductions** for the OT and x4_4 datasets
  (`experiments/` folder).
- **Documentation site** built with MkDocs Material, including a
  step-by-step algorithm walkthrough, per-step deep dives, and an
  auto-generated API reference.
- **Test suite** with 138 tests covering all algorithm steps,
  CI tests, the independent-change principle (with majority-vote
  correctness across multiple seeds), plotting, and evaluation.

### Notes

- This is a Python re-implementation. Results should match the paper
  qualitatively but are not bit-for-bit identical with the MATLAB
  reference, primarily because:
  - The Python KCI uses fixed-bandwidth kernels (the MATLAB reference
    uses GP-learned widths for `T <= 1000`).
  - The Python pipeline runs Step 1 (PCMCI) from scratch instead of
    using a hard-coded initial graph.
- The driving-force visualization (`cd_non_con_fun.m`) from the MATLAB
  reference is not yet ported.

[0.1.0]: https://github.com/hferdous/CDANs/releases/tag/v0.1.0
