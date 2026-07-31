"""advection_dataset — NodeDef-channel definition + providers."""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._pde_common import (
    materialize_pde,
    pde_def,
    preview_pde,
)
from comfy_research.nodes.registry import dataset_def, dataset_materializer_for, dataset_preview_for

ADVECTION = dataset_def(
    pde_def("advection_dataset", "Advection field (2D)", "AdvectionDatasetNode", hint="Constant-velocity advection on a periodic 2D grid; flattened trajectory windows.")
)

dataset_materializer_for(ADVECTION)(materialize_pde)
dataset_preview_for(ADVECTION)(preview_pde)
