"""hole_counting_dataset — NodeDef-channel definition.

程序化生成自 committed manifest。materialization uses a specialized branch
provider because cross_entropy_dense flattening depends on the task;
preview 无分支(独立 gallery API,capability 校验,family 字节保留零改动)。
seed 镜像方向与 token 家族相反:initSeed 是画布框 → seed sweepable=False。
"""
from __future__ import annotations

from comfy_research.nodes.registry import dataset_def
from comfy_research.nodes.schema import BoolField, DatasetDef, EnumField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="hole_counting_dataset",
        label="Hole counting (synthetic blobs)",
        hint="Synthetic blob with k circular holes; classify k (0..max holes).",
        family=("vision_dataset", "canvas_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            IntField(key="trainSize", label="Train Size", default=2048, min=1),
            IntField(key="testSize", label="Test Size", default=512, min=0),
            IntField(key="initSeed", label="Init Seed", default=0, min=0),
            IntField(key="seed", label="Seed", default=0, sweepable=False),
            BoolField(key="flattenOutput", label="Flatten Output", default=False),
            EnumField(key="samplingMode", label="Sampling Mode", default="fixed"),
            EnumField(key="specCodeName", label="Spec Code Name", default="hole_counting_datasetSpec"),
            IntField(key="imageSize", label="Image Size", default=48, min=24, max=96),
            IntField(key="maxHoles", label="Max Holes", default=3, min=0, max=8),
        ),
        frontend=FrontendSpec(component_key="VisionDatasetNode", codegen_key="hole_counting_dataset"),
    )
)
