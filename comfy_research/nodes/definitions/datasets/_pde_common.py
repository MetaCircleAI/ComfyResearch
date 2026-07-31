"""PDE field dataset 家族的共享 provider 体。

顶层 stdlib-only:engine/numpy import 全在函数体。
provider 体 = 原 family arm / preview 分支逐字节搬运,不重解释。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from comfy_research.nodes.schema import (
    DatasetDef,
    EnumField,
    Field,
    FloatField,
    FrontendSpec,
    IntField,
)

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.nodes.provider_types import DatasetMaterializeContext, DatasetPreviewRequest

PDE_FAMILY: tuple[str, ...] = (
    "pde_field_dataset",
    "vector_regression_dataset",
    "diffusion_noise_dataset",
    "canvas_dataset_source",
    "canvas_activation_dataset_source",
    "canvas_trainer_autoconnect_dataset",
)

# 14 字段逐字节对 manifest(裸四键);min 元数据只进 generatedNodeSpecs
# (取自 PdeFieldDatasetNode 的 widget 约束);samplingMode 无 options
# → 不生成 samplingMode enum sweep 轴。
def pde_fields() -> tuple[Field, ...]:
    return (
        IntField(key="contextFrames", label="Context Frames", default=4, min=1),
        IntField(key="channels", label="Channels", default=1, min=1),
        IntField(key="gridSize", label="Grid Size", default=16, min=4),
        IntField(key="trainSize", label="Train Size", default=512, min=1),
        IntField(key="testSize", label="Test Size", default=128, min=0),
        IntField(key="warmupSteps", label="Warmup Steps", default=40, min=0),
        FloatField(key="dt", label="Dt", default=0.05, min=1e-6),
        FloatField(key="diffusionCoeff", label="Diffusion Coeff", default=0.2, min=0),
        FloatField(key="reactionRate", label="Reaction Rate", default=1, min=0),
        FloatField(key="velocityX", label="Velocity X", default=0.5),
        FloatField(key="velocityY", label="Velocity Y", default=0.2),
        FloatField(key="icScale", label="Ic Scale", default=0.5, min=0),
        IntField(key="initSeed", label="Init Seed", default=0, min=0),
        EnumField(key="samplingMode", label="Sampling Mode", default="fixed"),
    )


def pde_def(type_: str, label: str, component_key: str, hint: str | None = None) -> DatasetDef:
    return DatasetDef(
        type=type_,
        label=label,
        hint=hint,
        family=PDE_FAMILY,
        fields=pde_fields(),
        frontend=FrontendSpec(component_key=component_key, codegen_key=type_),
    )


def materialize_pde(ctx: "DatasetMaterializeContext"):
    """原 family arm 体(dataset_materialize.py PDE arm, 搬运)。"""
    from comfy_research.engine.datasets.pde_field_dataset_runtime import build_pde_field_arrays
    from comfy_research.engine.trainer.dataset_materialize import _materialize_paired_split

    return _materialize_paired_split(
        lambda: build_pde_field_arrays(ctx.ds_type, ctx.rng, ctx.dd_train, ctx.dd_test, ctx.train_size, ctx.test_size),
        ctx.ds_type, ctx.rng, ctx.dd_train, ctx.dd_test, ctx.train_size, ctx.test_size,
        ctx.input_dim, ctx.output_dim, "Internal: PDE field test tensors missing.",
    )


def preview_pde(req: "DatasetPreviewRequest"):
    """原 _build_pde_field_arrays_tensor 体(dataset_tensor.py, 搬运)。"""
    import numpy as np
    from fastapi import HTTPException

    from comfy_research.engine.datasets.pde_field_dataset_runtime import build_pde_field_arrays, pde_field_flat_dims
    from comfy_research.engine.trainer.scalar import _scalar_int
    from comfy_research.schemas.graph import NodeKind

    kind_map = {
        "diffusion_pde_dataset": NodeKind.diffusion_pde_dataset,
        "reaction_diffusion_dataset": NodeKind.reaction_diffusion_dataset,
        "advection_dataset": NodeKind.advection_dataset,
    }
    data = req.data
    nk = kind_map.get(req.effective_type)
    if nk is None:
        raise HTTPException(status_code=400, detail=f"Not a PDE field dataset type: {req.effective_type}.")
    train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
    test_sz = max(0, _scalar_int(data.get("testSize"), 200))
    rng = np.random.default_rng(_scalar_int(data.get("initSeed"), 0))
    x_train, y_train, x_test, y_test = build_pde_field_arrays(nk, rng, data, data, train_sz, test_sz)
    _, _, _, flat = pde_field_flat_dims(data)
    if x_test is None:
        x_test = np.zeros((0, flat), dtype=np.float32)
    if y_test is None:
        y_test = np.zeros((0, flat), dtype=np.float32)
    return x_train, y_train, x_test, y_test
