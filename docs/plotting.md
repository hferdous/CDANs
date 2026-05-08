# Plotting

CDANs ships two plotting functions, both wrappers around `tigramite.plotting`:

* [`plot_process_graph`](api/plotting.md#cdans.plotting.plot_process_graph) —
  one node per variable, with curved arrows for lagged edges and straight
  arrows for contemporaneous ones. The surrogate `C` is materialized as
  an extra node connected to changing modules.
* [`plot_time_series_graph`](api/plotting.md#cdans.plotting.plot_time_series_graph) —
  time-unrolled, one node per `(variable, time-step)`. The surrogate `C`
  is drawn as an extra row at the top (by default), with arrows down to
  changing modules at every time-step. Changing modules are also
  highlighted in orange.

## Install

Plotting is an optional extra:

```bash
pip install "cdans[viz]"
```

This brings in `matplotlib` and `tigramite`. The core library does not
require either at runtime — only when you call a plotting function.

## Quick example

```python
from cdans import CDANs, plot_process_graph, plot_time_series_graph
from cdans.utils import generate_synthetic_cdans

dataset = generate_synthetic_cdans(n_vars=5, n_samples=400, tau_max=2, seed=0)
result = CDANs(tau_max=2, ci_test="kci").fit(dataset.data)

# Process graph (aggregate, with surrogate C drawn explicitly)
plot_process_graph(
    result.graph,
    var_names=["HR", "BP", "SpO2", "RR", "Temp"],
    save_path="process.png",
)

# Time-series graph (time-unrolled, changing modules in orange)
plot_time_series_graph(
    result.graph,
    var_names=["HR", "BP", "SpO2", "RR", "Temp"],
    save_path="time_series.png",
)
```

## Customizing the plot

Both functions accept the same `**kwargs` as their tigramite counterparts.
A few that are worth knowing:

| Argument               | Purpose                                          |
| ---------------------- | ------------------------------------------------ |
| `figsize=(w, h)`       | Matplotlib figure size in inches                 |
| `arrow_linewidth=4`    | Edge thickness                                   |
| `node_size=0.4`        | Node radius (process graph)                      |
| `label_fontsize=12`    | Variable-name font size                          |
| `fig_ax=(fig, ax)`     | Draw into an existing matplotlib axis            |
| `save_path="out.png"`  | Save instead of just returning the figure        |
| `show_surrogate=False` | Hide the `C` node entirely                       |
| `surrogate_position="bottom"` | Move the `C` row from top to bottom (time-series graph only) |

For the process graph specifically, `show_surrogate=False` hides the `C`
node if you'd rather see only the data variables.

## Output format

Both functions return a `(fig, ax)` matplotlib pair so you can post-process
or embed in a larger figure. The `save_path` keyword is a convenience that
calls `fig.savefig(save_path, bbox_inches="tight", dpi=150)`.

## What the link types mean

In tigramite's plotting convention:

* **Solid straight arrow** between two nodes: directed contemporaneous
  edge (e.g. `X1 -> X2`).
* **Empty-circle endpoints** (small unfilled circles instead of arrowheads):
  undirected contemporaneous edge — the algorithm couldn't determine a
  direction.
* **Curved arrow** with a number label: lagged edge; the number is the lag.
* **Edges from `C`**: the surrogate variable's effect on changing modules.
  Visible in both views by default; turn off with `show_surrogate=False`.
