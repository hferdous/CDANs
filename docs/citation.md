# Citation

If you use this library in your research, please cite the original paper:

> Ferdous, M. H., Hasan, U., & Gani, M. O. (2023). *CDANs: Temporal Causal
> Discovery from Autocorrelated and Non-Stationary Time Series Data.*
> In *Proceedings of the 8th Machine Learning for Healthcare Conference*
> (PMLR Vol. 219). [Paper](https://proceedings.mlr.press/v219/ferdous23a.html)

## BibTeX

```bibtex
@inproceedings{ferdous2023cdans,
  title     = {{CDANs}: Temporal Causal Discovery from Autocorrelated and
               Non-Stationary Time Series Data},
  author    = {Ferdous, Muhammad Hasan and Hasan, Uzma and Gani, Md Osman},
  booktitle = {Proceedings of the 8th Machine Learning for Healthcare Conference},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  volume    = {219},
  year      = {2023},
  url       = {https://proceedings.mlr.press/v219/ferdous23a.html}
}
```

## Citing this implementation specifically

A `CITATION.cff` file is shipped at the repo root for tools that
read it (e.g., GitHub's "Cite this repository" widget). The library
itself is MIT-licensed; see `LICENSE` in the repo.

## Related work

The CDANs algorithm builds on:

- **PCMCI** for lagged-adjacency discovery — Runge et al. (2019), *Detecting
  and quantifying causal associations in large nonlinear time series datasets.*
  *Science Advances*, 5(11).
- **CD-NOD** for non-stationary causal discovery — Huang, Zhang, Zhang,
  Ramsey, Sanchez-Romero, Glymour, Schölkopf (2020), *Causal Discovery from
  Heterogeneous/Nonstationary Data.* *JMLR* 21.
- **The independent-change principle** — Huang, Zhang, Sanchez-Romero,
  Ramsey, Glymour, Glymour (2017), *Behind Distribution Shift: Mining Driving
  Forces of Changes and Causal Arrows.*
- **PC-stable** for order-independent skeleton selection — Colombo &
  Maathuis (2014), *Order-Independent Constraint-Based Causal Structure
  Learning.* *JMLR* 15.
- **KCI** as a kernel-based CI test — Zhang, Peters, Janzing, Schölkopf
  (2011), *Kernel-based Conditional Independence Test and Application in
  Causal Discovery.* *UAI*.
- **Meek's rules** for orientation propagation — Meek (1995), *Causal
  inference and causal explanation with background knowledge.* *UAI*.
