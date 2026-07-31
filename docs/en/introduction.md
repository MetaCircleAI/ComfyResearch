---
doc_type: overview
doc_status: stable-core
---

:::{div} cr-eyebrow
Introduction
:::

# Comfy Research

:::{div} cr-lead
Compose an experiment as a node graph. Run it, inspect how learning changes,
and keep the complete experimental structure reproducible.
:::

::::{div} cr-intro-actions
:::{button-ref} get-started/index
:color: primary
:class: cr-intro-action

Start on CPU · 10–15 min
:::

:::{button-ref} get-started/first-graph
:color: primary
:outline:
:class: cr-intro-action

Already installed? Run the first graph
:::

:::{button-link} https://online.comfy-research.com/
:color: primary
:class: cr-intro-action

Try online ↗
:::
::::

Comfy Research is a local visual environment for AI/ML research. A graph
connects the parts of an experiment: data, model, objective, optimizer, and
training procedure, with Observables that measure what happens during learning.

The graph is more than a picture. It is the experiment specification sent to
the backend: node settings become parameters, edges determine dependencies,
and the Trainer validates the connected graph before starting a run.

```{figure} _images/app/overview-stable-workbench.png
:alt: Comfy Research workbench showing a project, a connected training graph, Trainer controls, and an Observable result.
:class: cr-product-figure

One graph keeps the experiment structure, execution controls, measurements,
and recorded result visible in the same workbench.
```

## Why experiments are graphs

Research code usually mixes three different concerns in one script:

1. **What is being trained**: the dataset, model, loss, and optimizer.
2. **How it is being measured**: loss curves, accuracy, norms, spectra, or
   representations.
3. **What must be retained**: configuration, results, checkpoints, and the
   relationship between them.

Comfy Research keeps those concerns visible and composable. Changing a model
or attaching a new measurement becomes a graph edit rather than a rewrite of
the whole training script.

## The product model

Every standard training graph has the same center of gravity:

| Part | Responsibility |
| --- | --- |
| Dataset | Produces the train and optional test data. |
| Model | Defines the trainable computation. |
| Loss | Defines the primary optimization objective. |
| Optimizer | Updates model parameters; schedules can be connected separately. |
| Trainer | Validates the graph, runs training, and streams progress and results. |
| Observable | Reads training state at log steps without changing the optimization objective. |
| Visualization | Displays loss or Observable histories produced by the Trainer. |

This separation matters experimentally. An Observable can be added to inspect
a hypothesis without silently changing the loss being optimized.

## From question to reproducible artifact

A normal workflow is:

1. Start from a small graph or a bundled template.
2. Change one controlled part of the experiment.
3. Run locally on CPU, MPS, or CUDA, or use the configured remote GPU path.
4. Read the Trainer and Observable outputs.
5. Save the graph at the level needed for reproduction.

Comfy Research supports reusable **Templates** as starting points.

Graph files and library entries can be saved as structure only, with plots, or
with model checkpoint data. The [User Guide](user-guide/index.md) explains the
trade-offs.

## Documentation scope

This documentation covers the stable research path in the current source
tree: the canvas, node graph, Trainer, Observables, saved graphs, and the
in-repository Node extension system. Experimental product areas are
intentionally outside this first documentation release.

:::::{grid} 1 1 2 2
:gutter: 3
:class-container: cr-link-grid

::::{grid-item-card} Get Started
:link: get-started/index
:link-type: doc
:class-card: cr-link-card
Build from source, open the canvas, and complete a CPU training run.
::::

::::{grid-item-card} User Guide
:link: user-guide/index
:link-type: doc
:class-card: cr-link-card
Understand graph wiring, runs, Observables, persistence, and reproducibility.
::::

::::{grid-item-card} Examples
:link: examples/index
:link-type: doc
:class-card: cr-link-card
Reproductions of classic learning-mechanics and physics-of-AI phenomena.
::::

::::{grid-item-card} Extend
:link: extend/index
:link-type: doc
:class-card: cr-link-card
Define and register custom Nodes and Observables in the repository.
::::

:::::
