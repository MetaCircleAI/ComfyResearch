"""reaction_diffusion_dataset — NodeDef-channel definition + providers."""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._pde_common import (
    materialize_pde,
    pde_def,
    preview_pde,
)
from comfy_research.nodes.registry import dataset_def, dataset_materializer_for, dataset_preview_for

REACTION_DIFFUSION = dataset_def(
    pde_def("reaction_diffusion_dataset", "Reaction–diffusion field (Fisher–KPP)", "ReactionDiffusionDatasetNode", hint="Fisher–KPP reaction–diffusion on a periodic grid; paired targets one Euler step ahead.")
)

dataset_materializer_for(REACTION_DIFFUSION)(materialize_pde)
dataset_preview_for(REACTION_DIFFUSION)(preview_pde)
