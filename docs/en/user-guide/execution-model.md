---
doc_type: explanation
doc_status: stable-core
---

:::{div} cr-eyebrow
User Guide
:::

# Understand graph execution

:::{div} cr-article-lead
The canvas is an editor for a typed experiment document; the backend executes
the serialized dependencies, not pixels or visual proximity.
:::

## From canvas to request

A graph document contains nodes, edges, and viewport state. Each node has a
type, identifier, position, and data fields. Each edge identifies its source,
target, source handle, and target handle.

When a Trainer starts, the frontend serializes the execution graph rooted in
that Trainer. The backend validates required inputs and resolves the connected
dataset, model, optimizer, loss, schedules, and Observables before allocating
the run.

## Training and measurement are separate

The primary loss participates in backpropagation. Observables read permitted
training state at log points. Connecting an Observable does not add it to the
loss, although computing it can make the run slower.

The Trainer returns distinct logical outputs:

- a model checkpoint for downstream checkpoint consumers;
- loss and test histories for Training viz;
- named Observable histories for their paired visualizations.

That separation makes it possible to change a measurement without silently
changing the objective.

## State has more than one lifetime

Node settings and graph structure belong to the graph document. Workspace
organization belongs to the workspace snapshot. Result histories and model
bytes are runtime state that may be removed by a smaller save tier. Remote
credentials live outside the graph under `.comfyresearch/`.

Therefore, a visible canvas is not automatically a complete reproducibility
bundle. Choose a save tier and external method record that match the claim.

## Events are incremental

Training is streamed as newline-delimited JSON. Progress arrives before a
terminal completed, paused, aborted, or error outcome. The frontend commits
histories and checkpoint state back into the graph only as those events are
handled.

See [Training API](../reference/training-api.md) for the supported route and
event contracts.
