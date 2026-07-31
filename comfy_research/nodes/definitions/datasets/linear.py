"""linear_dataset — NodeDef-channel definition + providers.

Canvas add 有 userLinearDatasetId options 线程 → generic add 分支显式排除本 type
(observable_user 先例),特殊分支保留。
"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._linear_common import (
    LINEAR_FAMILY,
    materialize_linear_like,
    preview_direct_arrays,
    regression_fields,
)
from comfy_research.nodes.registry import dataset_def, dataset_materializer_for, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, FrontendSpec

LINEAR = dataset_def(
    DatasetDef(
        type="linear_dataset",
        label="Linear Dataset",
        hint="Synthetic linear regression dataset feeding the MLP trainer.",
        family=LINEAR_FAMILY,
        fields=regression_fields(
            input_dim=10, output_dim=1, out_dist_default="additive_gaussian",
            train_size=800, test_size=200, noise_level=0.25, seed=0,
        ),
        frontend=FrontendSpec(component_key="LinearDatasetNode", codegen_key="linear_dataset"),
    )
)

dataset_materializer_for(LINEAR)(materialize_linear_like)
dataset_preview_for(LINEAR)(preview_direct_arrays)
