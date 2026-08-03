---
doc_type: tutorial
doc_status: stable-core
template_id: edge-of-stability-cpu
---

:::{div} cr-eyebrow
Get Started
:::

# Run your first graph

:::{div} cr-article-lead
Load a complete experiment, run it on CPU, verify the recorded result, and
change one variable without rewriting a training script.
:::

## Before you begin

Complete [Install and run](index.md), keep the backend running, and confirm
`http://127.0.0.1:8042/api/health` returns `"ok": true`.

This tutorial uses the bundled `Edge of Stability (CPU)` template. It contains a synthetic
linear dataset, an MLP, MSE loss, SGD, a Trainer, and an Observable. The run is
small enough for CPU and does not download a dataset.

## 1. Load the template

1. Select **Templates** in the left rail.
2. Open **Edge of Stability (CPU)**.
3. Wait for its project and saved graph to appear on the canvas.

:::{figure} ../_images/app/first-graph-template.png
:alt: Templates view with the recommended first-run graph template selected.
:class: cr-product-screenshot
:::

## 2. Read the graph before running it

Find the **Trainer** node and confirm that **compute device** is **CPU**. Its
Dataset, MLP, MSE, and SGD inputs should already be connected. The graph itself
is the executable specification: node fields become parameters and edges tell
the backend which values satisfy each Trainer input.

The template is configured for 80 steps. It records every step so the
edge-of-stability dynamics stay visible; note its `trainingSteps`,
`logFrequency`, and `batchSize` values before changing anything.

:::{figure} ../_images/app/first-graph-trainer.png
:alt: A connected Trainer node configured to run the example graph on CPU.
:class: cr-product-screenshot
:::

## 3. Run and inspect the result

1. Click **Train** in the Trainer header.
2. Watch the Trainer progress state while the backend executes the graph.
3. When the run completes, inspect **Training viz** and the paired Observable
   visualization.
4. Check that the recorded loss has a history rather than a single value.

This is a qualitative CPU demonstration of the edge-of-stability (EoS) result
in [Damian, Nichani, and Lee (2023)](https://arxiv.org/abs/2209.15594), not a
numerical reproduction of its CIFAR-10 experiments. The Hessian visualization
uses its dashed `2/η = 10` line as the stability cutoff. In a successful run:

- `λ₁` rises to the cutoff, briefly overshoots it, then fluctuates around it;
- `λ₂` remains well below the cutoff, so the visible instability is confined
  to the top direction; and
- the loss can rise temporarily after the cutoff, but falls overall.

The Hessian Observable demonstrates that measurements are graph components.
It records at the Trainer's logging cadence and does not replace the loss
being optimized. Remove it later when a faster loss-only smoke test is more
useful.

:::{figure} ../_images/app/first-graph-results.png
:alt: Completed example run with a recorded Observable displayed as a chart.
:class: cr-product-screenshot
:::

## Success checkpoints

The first run is complete when all of these are true:

- the Trainer reaches the end of its 80 steps without an error state;
- Training viz contains a loss history;
- the paired Observable visualization contains `λ₁` and `λ₂`, plus its
  `2/η = 10` reference line;
- the graph is still editable after the run.

The qualitative relationship is the scientific checkpoint; exact crossing
steps and amplitudes are not universal pass criteria because EoS is sensitive
to numerical perturbations. The paper uses a substantially larger setup and
higher precision after instability, whereas this graph is deliberately small
enough for local CPU execution.

## Make one controlled change

Change the SGD learning rate from `0.2` to `0.1`, but leave the other settings
unchanged. Run the Trainer again and compare the curves: the cutoff becomes
`2/η = 20`, which the top eigenvalue should not reach in this short CPU demo,
and the loss should be more nearly monotonic. This is the negative control for
the EoS interpretation.

That small action is the core research loop:

1. keep the experimental structure visible;
2. change one variable;
3. record the resulting dynamics;
4. preserve enough context to explain the comparison.

When the graph is useful, save a **Small** template. Small retains graph
structure and settings while excluding plot histories and checkpoint bytes.
The [Projects and artifacts](../user-guide/projects-and-artifacts.md) guide
explains the larger save tiers and their costs.

Next, learn how to [build and run graphs](../user-guide/build-and-run-graphs.md)
without starting from a complete template.
