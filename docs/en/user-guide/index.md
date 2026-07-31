---
doc_type: overview
doc_status: stable-core
---

:::{div} cr-eyebrow
User Guide
:::

# Work with experiments

:::{div} cr-article-lead
Build a valid training graph, choose measurements deliberately, preserve the
right artifact, and diagnose failures from the Trainer outward.
:::

Start with the task you are trying to complete. Concept pages explain why the
application behaves that way without repeating the same sequence of clicks.

::::{grid} 1 1 2 2
:gutter: 3
:class-container: cr-link-grid

:::{grid-item-card} Build and run graphs
:link: build-and-run-graphs
:link-type: doc
:class-card: cr-link-card
Assemble the stable training core, validate its connections, and run a focused parameter series.
:::

:::{grid-item-card} Record Observables
:link: observables
:link-type: doc
:class-card: cr-link-card
Attach measurements without confusing them with the optimization objective.
:::

:::{grid-item-card} Manage projects and artifacts
:link: projects-and-artifacts
:link-type: doc
:class-card: cr-link-card
Organize canvases and choose Small, Medium, or Large persistence intentionally.
:::

:::{grid-item-card} Use a remote GPU
:link: remote-gpu
:link-type: doc
:class-card: cr-link-card
Configure SSH execution, verify the target, and keep credentials out of research artifacts.
:::

:::{grid-item-card} Make a result reproducible
:link: reproducibility
:link-type: doc
:class-card: cr-link-card
Record the graph, environment, seeds, measurements, and claim boundary.
:::

:::{grid-item-card} Troubleshoot a run
:link: troubleshooting
:link-type: doc
:class-card: cr-link-card
Work from visible symptoms to graph, device, service, or measurement causes.
:::

:::{grid-item-card} Understand graph execution
:link: execution-model
:link-type: doc
:class-card: cr-link-card
Learn how graph fields and edges become a validated backend request and streamed result.
:::

::::

```{toctree}
:hidden:

Build and run graphs <build-and-run-graphs>
Record Observables <observables>
Manage projects and artifacts <projects-and-artifacts>
Use a remote GPU <remote-gpu>
Make a result reproducible <reproducibility>
Troubleshoot a run <troubleshooting>
Understand graph execution <execution-model>
```
