"""random_noise_dataset — NodeDef-channel definition + providers.

字段定义同时提供 sweep 轴。
"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._linear_common import (
    RANDOM_NOISE_FAMILY,
    materialize_linear_like,
    preview_direct_arrays,
    regression_fields,
)
from comfy_research.nodes.registry import dataset_def, dataset_materializer_for, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, FrontendSpec

RANDOM_NOISE = dataset_def(
    DatasetDef(
        type="random_noise_dataset",
        label="Random noise dataset",
        family=RANDOM_NOISE_FAMILY,
        fields=regression_fields(
            input_dim=10, output_dim=1, out_dist_default="deterministic",
            train_size=800, test_size=200, noise_level=0, seed=0,
        ),
        frontend=FrontendSpec(component_key="LinearDatasetNode", codegen_key="random_noise_dataset"),
    )
)

dataset_materializer_for(RANDOM_NOISE)(materialize_linear_like)
dataset_preview_for(RANDOM_NOISE)(preview_direct_arrays)
