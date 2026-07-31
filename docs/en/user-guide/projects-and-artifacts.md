---
doc_type: how-to
doc_status: stable-core
---

:::{div} cr-eyebrow
User Guide
:::

# Manage projects and artifacts

:::{div} cr-article-lead
Keep controlled variants separate and persist only the runtime state required
for the next use of the experiment.
:::

## Projects and canvases

A workspace contains projects, and each project owns one canvas. The canvas
stores a graph document and its viewport. Use a separate project for a
controlled variant instead of overwriting the only graph that explains an
earlier result.

Saved Templates open as working copies in new projects. The library entry
remains separate from later edits on the canvas.

:::{figure} ../_images/app/project-canvas-tree.png
:alt: Baseline and Comparison project tabs with the Baseline project's single canvas open in the workbench.
:class: cr-product-screenshot
:::

## Choose a destination

Use the Graph menu to save a graph file or Template:

- a **Template** is a reusable starting point.

## Choose a size tier

| Tier | Retains | Removes | Recommended use |
| --- | --- | --- | --- |
| Small | Graph structure and node settings | Plot histories and checkpoint bytes | Reviewable templates and graph files |
| Medium | Graph and plot or visualization data | Checkpoint bytes | Sharing completed curves without a model blob |
| Large | Full graph, plots, and model checkpoint data | Nothing | Local archival when trained state is required |

:::{figure} ../_images/app/save-artifact-tiers.png
:alt: Save dialog showing the available artifact persistence tiers.
:class: cr-product-screenshot
:::

Prefer Small for committed Templates. Medium and Large JSON can
contain experiment results or encoded model state, grow rapidly, and expose
information that does not belong in Git history. Inspect a saved file before
sharing it.

## Know what is local state

The active workspace is persisted through `/api/workspace` to
`data/workspace.json`. Graph-library data is stored under
`data/graph_library/`; committed canonical templates live in
`data/graph_library/templates/`.

The repository ignores most local workspace, library, credential, and runtime
artifact state, but ignore rules are not a data-handling policy. Check the
actual file and Git status before committing or sending an archive.

Use [Make a result reproducible](reproducibility.md) to decide what context a
shared artifact still needs outside its JSON.
