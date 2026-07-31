"""cifar10_dataset — NodeDef-channel definition.

字段/默认值转写自 repro-reference manifest 条目(镜像 mnist_dataset:
seed 是镜像字段 → sweepable=False;materialize 走 vision_dataset capability
分派 + build_vision_numpy_arrays 的 cifar10 分支,零 provider)。
"""
from __future__ import annotations

from comfy_research.nodes.registry import dataset_def
from comfy_research.nodes.schema import BoolField, DatasetDef, EnumField, FloatField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="cifar10_dataset",
        label="CIFAR-10 dataset",
        hint="Official CIFAR-10 (32×32 RGB): bundled data/cifar10 or URL fallback; 10-class labels.",
        family=(
            "vision_dataset",
            "diffusion_noise_dataset",
            "canvas_dataset_source",
            "canvas_trainer_autoconnect_dataset",
        ),
        fields=(
            IntField(key="trainSize", label="Train Size", default=2048, min=1),
            IntField(key="testSize", label="Test Size", default=512, min=0),
            IntField(key="initSeed", label="Init Seed", default=0, min=0),
            IntField(key="seed", label="Seed", default=0, sweepable=False),
            IntField(key="subsetSeed", label="Subset Seed", default=0, min=0),
            BoolField(key="classBalanced", label="Class Balanced", default=True),
            EnumField(
                key="normalize",
                label="Normalize",
                default="zero_one",
                options=("zero_one", "minus_one_to_one"),
            ),
            EnumField(
                key="inputTransform",
                label="Input Transform",
                default="none",
                options=("none", "shuffled_pixels", "random_pixels", "gaussian"),
            ),
            EnumField(
                key="preprocessing",
                label="Preprocessing",
                default="none",
                options=("none", "center_crop_28_per_image_whiten"),
            ),
            FloatField(key="labelCorruption", label="Label Corruption", default=0.0, min=0.0, max=1.0),
            EnumField(
                key="trainingRecipe",
                label="Training Recipe",
                default="standard",
                options=("standard", "jastrzbski_fig1"),
                sweepable=False,
            ),
            BoolField(key="flattenOutput", label="Flatten Output", default=False),
            EnumField(key="samplingMode", label="Sampling Mode", default="fixed"),
            EnumField(key="specCodeName", label="Spec Code Name", default="cifar10_datasetSpec"),
            EnumField(key="downloadCacheDir", label="Download Cache Dir", default=""),
        ),
        frontend=FrontendSpec(component_key="VisionDatasetNode", codegen_key="cifar10_dataset"),
    )
)
