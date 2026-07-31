---
doc_type: reference
doc_status: stable-core
---

:::{div} cr-eyebrow
Reference
:::

# Data contracts

## Graph document

Graph document version: `1`.

```json
{
  "version": 1,
  "nodes": [],
  "edges": [],
  "viewport": { "x": 0, "y": 0, "zoom": 1 }
}
```

A node requires `id` and a generated `type`; it also carries `position`,
validated `data`, and optional parent-layout fields. An edge requires `id`,
`source`, and `target`, with nullable `sourceHandle` and `targetHandle`.

Only serialized edges determine dependencies. Position and viewport are editor
state and do not connect nodes.

Source: `comfy_research/schemas/graph.py`.

## Workspace snapshot

Workspace snapshot version: `3`.

The root contains `active_project_id` and one or more projects. Each project
owns exactly one canvas. `active_project_id` is validated against the project
collection. To compare variants, use one project per variant.

`GET /api/workspace` loads `data/workspace.json` when it exists. If it is
absent, the backend creates and persists a fresh version-3 workspace with one
project and one canvas. An invalid existing workspace is an error rather than
an instruction to silently discard state.

Source: `comfy_research/schemas/workspace.py` and
`comfy_research/api/workspace.py`.

## Saved graph entry

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Stable library identifier |
| `name` | string | Display name |
| `tier` | `small`, `medium`, or `large` | Export filtering already applied to `document` |
| `document` | GraphDocument | Saved graph version and content |
| `savedAt` | number | Unix time in milliseconds |
| `libraryOrigin` | optional `combined_model` | Origin marker for supported combined-model entries |

Libraries accept at most 200 entries. Workflow JSON lists keep the newest 200
inserts. Canonical Templates use one JSON file per ID under
`data/graph_library/templates/` and trim the oldest valid entries when the same
limit is exceeded.

Template IDs must be non-empty and cannot contain slash, backslash, or a
leading dot when used as canonical filenames.

## Export tiers

| Tier | Filtering contract |
| --- | --- |
| Small | Remove checkpoint bytes and known plot, visualization, and runtime histories |
| Medium | Remove checkpoint bytes but retain plot and visualization data |
| Large | Return the full graph document unchanged |

The tier describes filtering, not confidentiality. Inspect all documents before
publishing them.
