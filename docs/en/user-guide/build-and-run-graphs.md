---
doc_type: how-to
doc_status: stable-core
---

:::{div} cr-eyebrow
User Guide
:::

# Build and run graphs

:::{div} cr-article-lead
Create the smallest complete training graph, verify every dependency, and add
variation only after one configuration runs successfully.
:::

## What a runnable graph needs

A standard training experiment centers on one Trainer. Add these sources from
the **Nodes** rail:

| Trainer input | Source | Required |
| --- | --- | --- |
| `dataset` | Dataset node | Yes |
| `model` | Trainable model or combined model | Yes |
| `optimizer` | Optimizer node | Yes |
| `loss` | Compatible primary loss | Yes |
| `observables` | One or more Observable nodes | No |
| `batch sched` | Batch schedule | No |

:::{figure} ../_images/app/add-node-from-library.png
:alt: Node library search beside a canvas where the selected node can be added.
:class: cr-product-screenshot cr-product-screenshot-portrait
:::

## Connect the training core

1. Drag a Dataset, Model, Optimizer, Loss, and Trainer onto the canvas.
2. Connect each source socket to the matching Trainer input.
3. Add **Training viz** and connect the Trainer `loss` output to it.
4. Set the Trainer to a short CPU run.
5. Read the graph from each required source into the Trainer before clicking
   **Train**.

The connection handles are typed. A line that looks close to an input is not
enough: the serialized edge must carry the intended `sourceHandle` and
`targetHandle`. The backend rejects missing inputs and incompatible graph
components before running a partial experiment.

:::{figure} ../_images/app/stable-training-core.png
:alt: Complete training graph connecting data, model, optimizer, loss, and Trainer nodes.
:class: cr-product-screenshot
:::

## Configure one run

Start with values that are cheap to inspect:

- use steps rather than epochs until dataset size and batching are understood;
- use CPU unless the graph specifically requires an accelerator;
- choose a log frequency that yields enough points to diagnose the curve;
- use `-1` batch size only when full-batch training is intentional;
- leave gradient clipping at zero unless clipping is part of the method.

Click **Train**. During a local run the Trainer can request pause or abort. A
completed run sends the model checkpoint, loss histories, and Observable
histories through separate output channels.

## Add a parameter series carefully

Supported numeric controls accept comma-separated values. When more than one
field has multiple values, Comfy Research builds the Cartesian product of the
sweep axes. Two learning rates and three batch sizes therefore produce six
runs, not three.

Begin with one varying axis and hold seeds, data, measurements, and training
length fixed. Verify that its runs are interpretable before adding a second
axis. The application caps a train series at 256 combinations, but a smaller
scientifically justified sweep is usually better than using the technical
maximum.

## Verify the result

Before saving, confirm that:

- the Trainer reached its terminal completed state;
- Training viz received the expected number of logged points;
- every attached Observable produced the intended result channel;
- the graph still shows the exact configuration that produced the result;
- every sweep axis and fixed control is identifiable.

Next, decide which measurements belong in the graph in
[Record Observables](observables.md).
