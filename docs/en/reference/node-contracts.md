---
doc_type: reference
doc_status: stable-core
---

:::{div} cr-eyebrow
Reference
:::

# Node contracts

:::{div} cr-article-lead
Definition sources, runtime registration channels, generated artifacts, and
validation commands for the in-repository Node API.
:::

## Contract pipeline

A Node travels through four layers:

1. A frozen `*Def` object declares its type, label, category, fields,
   capabilities, frontend component, and optional ports.
2. A registry function records the definition and, when applicable, a runtime
   provider such as a model builder or Observable recorder.
3. `comfy_research/nodes/generate.py` produces backend and frontend contracts.
4. The frontend uses generated specs to render the Node while the backend uses
   the registered provider when the graph executes.

Definitions under `comfy_research/nodes/definitions/` are discovered
recursively. Adding a module under the correct package does not require a
central import list.

## Node channels

| Node kind | Definition | Runtime registration |
| --- | --- | --- |
| Dataset | `DatasetDef` via `dataset_def(...)` | Optional `dataset_materializer_for(...)` and/or `dataset_preview_for(...)` |
| Model | `ModelDef` via `model_def(...)` | `model_builder_for(...)` for runnable full models |
| Loss | `LossDef` via `loss_def(...)` | `loss_criterion_for(...)` for primary criteria |
| Optimizer | `OptimizerDef` via `optimizer_def(...)` | `optimizer_builder_for(...)` |
| Trainer | `TrainerDef` via `trainer_def(...)` | Uses the Trainer preparation and execution pipeline |
| Initialization | `InitializationDef` via `initialization_def(...)` | Applied by the Trainer runtime |
| Observable | `ObservableDef` via `observable_def(...)` | `recorder_for(...)` is required |
| Frontend-only | `FrontendNodeDef` via `frontend_node_def(...)` | No backend provider |

## Definition sources

| Source | Responsibility |
| --- | --- |
| `comfy_research/nodes/schema.py` | Definition dataclasses, field types, ports, frontend metadata, and Observable metadata |
| `comfy_research/nodes/registry.py` | Provider decorators, discovery, uniqueness, and completeness validation |
| `comfy_research/nodes/definitions/` | Dataset, model, loss, optimizer, trainer, initialization, Observable, and frontend definitions |
| `comfy_research/nodes/generate.py` | Deterministic backend and frontend contract generation |

Change a definition or generator when the contract changes. Do not edit
generated files to introduce behavior.

## Generated outputs

The generator writes these main artifacts:

- `comfy_research/generated/node_manifest.json` for served node metadata;
- `comfy_research/generated/node_kind.py` for accepted generated node kinds;
- `comfy_research/generated/node_capabilities.py` for capability lookups;
- `comfy_research/generated/node_params.py` for per-node parameter validation;
- `frontend/src/generated/generatedNodeSpecs.ts` for frontend defaults, fields,
  ports, categories, and sweep metadata.

The Python graph schema validates a node's `data` with the generated parameter
model for its type. The frontend build also checks generated metadata, so stale
artifacts fail before a production bundle is accepted.

## Generation and validation

Generate after changing definitions:

```bash
npm --prefix frontend run generate:node-manifest
```

Verify that the checked-in output exactly matches current definitions:

```bash
npm --prefix frontend run verify:node-manifest
```

Validate registry completeness directly:

```bash
python -c "from comfy_research.nodes.registry import validate_defs; validate_defs()"
```

`npm --prefix frontend run build` runs the Node manifest check together with
the other research-code and golden-contract verifiers.

## Edit policy

Do not edit generated files by hand. A manual change can be overwritten on the
next generation and can make the frontend, backend, and tests disagree about
accepted fields or connections. Review both the source definition and the
generated diff when changing a Node contract.

For a complete definition workflow, see
[Add a scalar Observable](../extend/add-observable.md).
