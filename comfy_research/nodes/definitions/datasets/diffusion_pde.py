"""diffusion_pde_dataset — NodeDef-channel definition + providers."""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._pde_common import (
    materialize_pde,
    pde_def,
    preview_pde,
)
from comfy_research.nodes.registry import dataset_def, dataset_materializer_for, dataset_preview_for

DIFFUSION_PDE = dataset_def(
    pde_def("diffusion_pde_dataset", "Diffusion PDE field (2D)", "DiffusionPdeDatasetNode", hint="Synthetic 2D heat-equation fields: flattened space-time windows for MSE regression / MPP-style ViT.")
)

dataset_materializer_for(DIFFUSION_PDE)(materialize_pde)
dataset_preview_for(DIFFUSION_PDE)(preview_pde)
