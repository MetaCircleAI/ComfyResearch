---
doc_type: tutorial
doc_status: stable-core
---

:::{div} cr-eyebrow
Extend
:::

# Add a scalar Observable

:::{div} cr-article-lead
Define one measurement, register its runtime recorder, regenerate the shared
contract, and verify the complete backend-to-frontend path.
:::

## Understand the extension boundary

The current extension mechanism is an in-repository API, not a runtime plugin
loader. A new Node is a code contribution under
`comfy_research.nodes.definitions` and is checked in with its generated
artifacts.

This tutorial uses an Observable because its declaration and runtime
measurement can live in one file. For every supported Node channel and its
registration contract, see [Node contracts](../reference/node-contracts.md).

## Define and register the Observable

Create
`comfy_research/nodes/definitions/observables/weight_mean_abs.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import (
    FrontendSpec,
    ObservableDef,
    SpawnSpec,
    VizSpec,
)

if TYPE_CHECKING:
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node


WEIGHT_MEAN_ABS = observable_def(
    ObservableDef(
        type="observable_weight_mean_abs",
        label="Weight mean absolute value",
        hint="Mean absolute value of all trainable parameters.",
        viz=VizSpec(
            variant="user",
            title="Weight mean absolute value",
            info_markdown="Mean absolute value across trainable parameters.",
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", unit="mean |w|"),
        ),
        frontend=FrontendSpec(),
    )
)


@recorder_for(WEIGHT_MEAN_ABS)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    import torch

    with torch.no_grad():
        parameters = [p.detach() for p in rec.model.parameters() if p.requires_grad]
        count = sum(p.numel() for p in parameters)
        total = sum(float(p.abs().sum().item()) for p in parameters)

    value = total / count if count else float("nan")
    rec.observable_metric_histories[on.id].append(value)
```

The important invariants are:

- `type` is globally unique and stable because saved graphs serialize it.
- `variant="user"` and `SpawnSpec(kind="user_scalar")` use the generic scalar
  visualization path, so no custom React component is needed.
- The recorder appends exactly one value for each call.
- Heavy imports stay inside the provider function so definition discovery and
  generation remain lightweight.
- `validate_defs()` requires a registered Observable to have one recorder and
  prevents a recorder from existing without its definition.

## Add fields when the measurement is configurable

Declare user-editable parameters in `fields`, for example:

```python
from comfy_research.nodes.schema import IntField

fields=(
    IntField(key="everyNthLayer", label="Every Nth layer", default=1, min=1),
)
```

The value is serialized under the node's `data` object and is available to the
recorder through `on.data`. Use the narrowest field type and an explicit
minimum when the UI can reject invalid values.

## Regenerate the contract

From the repository root:

```bash
npm --prefix frontend run generate:node-manifest
npm --prefix frontend run verify:node-manifest
```

Review every generated difference. The expected files and their responsibilities
are listed in [Node contracts](../reference/node-contracts.md).

## Test the extension

Run the definition and generation contracts:

```bash
python -m pytest -q \
  comfy_research/tests/test_nodes_generate_contract.py \
  comfy_research/tests/test_trainer_pkg_boundaries.py
```

Add a focused provider test that constructs the smallest recorder state, calls
`record(...)`, and asserts the exact history value. Then verify the frontend:

```bash
npm --prefix frontend run build
```

Open the app and check the complete path:

1. The Node appears under **Observables**.
2. It can be placed and connected to a Trainer.
3. The connection creates or accepts the expected visualization.
4. Training appends values at the configured log frequency.
5. Saving and reopening the graph preserves its type and fields.


## When a custom frontend is justified

`FrontendSpec()` selects the generic component. Keep it when fields, sockets,
and a standard scalar or series visualization are sufficient.

A custom React component is warranted only when the Node needs interaction or
visualization that the generated schema cannot express. In that case, add a
stable `component_key`, implement and register the component in the frontend,
and add a node-registry invariant test. Do not add a custom component only to
change spacing, labels, or basic numeric controls.

## Contribution checklist

- Start from current `main` and use a focused topic branch.
- Keep the new Node type and saved-graph compatibility stable.
- Regenerate artifacts; never hand-edit them.
- Test runtime behavior and frontend registration.
- Include a Small template when the Node is best understood in a complete graph.
- Explain the scientific purpose and limitations of the measurement.
