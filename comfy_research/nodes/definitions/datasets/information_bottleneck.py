"""Verified local information-bottleneck input table."""

from __future__ import annotations

from comfy_research.nodes.registry import dataset_def, dataset_materializer_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FrontendSpec, IntField


DEF = dataset_def(
    DatasetDef(
        type="information_bottleneck_dataset",
        label="Information bottleneck input table",
        hint=(
            "Deterministically samples the locally bundled binary F/y table. "
            "The asset is SHA-256 verified at runtime; its external provenance and licence are unverified."
        ),
        family=(
            "memorization_dataset",
            "activation_sample_dataset",
            "canvas_dataset_source",
            "canvas_activation_dataset_source",
            "canvas_trainer_autoconnect_dataset",
            "dataset_tensor_direct_arrays",
        ),
        fields=(
            IntField(key="inputDim", label="Input Dim", default=12, min=12, max=12, sweepable=False),
            IntField(key="outputDim", label="Output Dim", default=2, min=2, max=2, sweepable=False),
            IntField(key="trainSize", label="Train Size", default=3482, min=1, max=4096),
            IntField(key="testSize", label="Test Size", default=4096, min=0, max=4096),
            IntField(key="seed", label="Seed", default=0),
            EnumField(key="samplingMode", label="Sampling Mode", default="fixed", sweepable=False),
        ),
        frontend=FrontendSpec(component_key="GenericDatasetNode", codegen_key="information_bottleneck_dataset"),
    )
)


@dataset_materializer_for(DEF)
def materialize(ctx):
    from comfy_research.engine.datasets.information_bottleneck_dataset import build_information_bottleneck_arrays
    from comfy_research.engine.trainer.dataset_arrays import DatasetArrays

    x_np, y_np, x_test_np, y_test_np = build_information_bottleneck_arrays(
        ctx.rng,
        train_size=ctx.train_size,
        test_size=ctx.test_size,
        input_dim=ctx.input_dim,
    )
    return DatasetArrays(
        x_np=x_np,
        y_np=y_np,
        x_test_np=x_test_np,
        y_test_np=y_test_np,
        input_dim=12,
        output_dim=2,
    )
