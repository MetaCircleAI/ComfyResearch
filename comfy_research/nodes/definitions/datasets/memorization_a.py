"""memorization_a_dataset — NodeDef-channel definition + preview.

Materialization uses the task-dependent dense branch for cross_entropy_dense;
the preview provider does not own that dispatch. noiseLevel is hidden in the UI
and therefore is not sweepable.
"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._linear_common import (
    MEM_A_FAMILY,
    MEM_OUTPUT_DIST_OPTIONS,
    preview_direct_arrays,
    regression_fields,
)
from comfy_research.nodes.registry import dataset_def, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FrontendSpec

MEMORIZATION_A = dataset_def(
    DatasetDef(
        type="memorization_a_dataset",
        label="Memorization A dataset",
        hint="Memory 1 dataset A: continuous x, random class labels (cross-entropy memorization). Pairs with memorization B.",
        family=MEM_A_FAMILY,
        fields=regression_fields(
            input_dim=40, output_dim=40, out_dist_default="uniform_class_probs",
            train_size=160, test_size=0, noise_level=0, seed=0,
            out_dist_options=MEM_OUTPUT_DIST_OPTIONS,
            noise_axis=False, alpha_axis=True,
            extras=(EnumField(key="specCodeName", label="Spec Code Name", default="Memorization_A_Dataset"),),
        ),
        frontend=FrontendSpec(component_key="LinearDatasetNode", codegen_key="memorization_a_dataset"),
    )
)

dataset_preview_for(MEMORIZATION_A)(preview_direct_arrays)
