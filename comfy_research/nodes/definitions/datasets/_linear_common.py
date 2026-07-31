"""linear-like 家族共享 def/provider 体。顶层 stdlib-only。

Sweep 轴纪律:
- 需要枚举轴的 enum 声明 options(只进 generatedNodeSpecs,manifest 恒裸);
- UI 隐藏数值(linear/random_noise 的 alpha、mem_a 的 noiseLevel、trio 的隐藏项)
  → sweepable=False;
- mem_b 的 inputDim/outputDim 被 vocabSize 镜像写入 → sweepable=False
  防一次多值输入变 2^3 组合爆炸。
memorization_a/b 的 materialization 由 trainer_task 条件分支处理；
ai4science 三型使用 alias-only runtime，不注册 provider。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from comfy_research.nodes.schema import (
    EnumField,
    Field,
    FloatField,
    IntField,
)

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.nodes.provider_types import DatasetMaterializeContext, DatasetPreviewRequest

INPUT_DIST_OPTIONS: tuple[str, ...] = ("standard_normal", "uniform_neg1_1", "uniform_0_1")
OUTPUT_DIST_OPTIONS: tuple[str, ...] = ("additive_gaussian", "deterministic")
MEM_OUTPUT_DIST_OPTIONS: tuple[str, ...] = (
    "uniform_class_probs", "power_law_class_probs", "exponential_class_probs",
)

LINEAR_FAMILY: tuple[str, ...] = (
    "linear_like_dataset", "vector_regression_dataset", "diffusion_noise_dataset",
    "activation_sample_dataset", "canvas_dataset_source", "canvas_activation_dataset_source",
    "canvas_trainer_autoconnect_dataset", "dataset_tensor_direct_arrays",
)
RANDOM_NOISE_FAMILY: tuple[str, ...] = (
    "linear_like_dataset", "vector_regression_dataset", "diffusion_noise_dataset",
    "canvas_dataset_source", "canvas_activation_dataset_source",
    "canvas_trainer_autoconnect_dataset", "dataset_tensor_direct_arrays",
)
MEM_A_FAMILY: tuple[str, ...] = (
    "memorization_dataset", "activation_sample_dataset", "canvas_dataset_source",
    "canvas_activation_dataset_source", "canvas_trainer_autoconnect_dataset",
    "dataset_tensor_direct_arrays",
)
MEM_B_FAMILY: tuple[str, ...] = (
    "memorization_dataset", "token_classification_dataset", "vector_regression_dataset",
    "diffusion_noise_dataset", "activation_sample_dataset", "canvas_dataset_source",
    "canvas_activation_dataset_source", "canvas_trainer_autoconnect_dataset",
    "dataset_tensor_direct_arrays",
)
AI4SCIENCE_FAMILY: tuple[str, ...] = ("canvas_trainer_autoconnect_dataset",)


def regression_fields(
    *, input_dim: int, output_dim: int, out_dist_default: str,
    train_size: int, test_size: int, noise_level: float, seed: int,
    sampling_mode: str = "fixed",
    input_dist_axis: bool = True, out_dist_options: tuple[str, ...] = OUTPUT_DIST_OPTIONS,
    noise_axis: bool = True, alpha_axis: bool = False,
    extras: tuple[Field, ...] = (),
) -> tuple[Field, ...]:
    return (
        IntField(key="inputDim", label="Input Dim", default=input_dim, min=1),
        IntField(key="outputDim", label="Output Dim", default=output_dim, min=1),
        EnumField(
            key="inputDistribution", label="Input Distribution", default="standard_normal",
            options=INPUT_DIST_OPTIONS if input_dist_axis else (),
            sweepable=input_dist_axis,
        ),
        EnumField(key="outputDistribution", label="Output Distribution", default=out_dist_default, options=out_dist_options),
        IntField(key="trainSize", label="Train Size", default=train_size, min=1),
        IntField(key="testSize", label="Test Size", default=test_size, min=0),
        FloatField(key="noiseLevel", label="Noise Level", default=noise_level, min=0, sweepable=noise_axis),
        FloatField(key="alpha", label="Alpha", default=1, min=0, sweepable=alpha_axis),
        IntField(key="seed", label="Seed", default=seed),
        EnumField(key="samplingMode", label="Sampling Mode", default=sampling_mode),
    ) + extras


def materialize_linear_like(ctx: "DatasetMaterializeContext"):
    """原 _LINEAR_LIKE arm 体(dataset_materialize.py:221-225 + 标量解析行原样)。"""
    from comfy_research.engine.trainer.dataset_materialize import _materialize_linear_like
    from comfy_research.engine.trainer.scalar import _scalar_float, _scalar_str

    dd_train, dd_test = ctx.dd_train, ctx.dd_test
    input_dist = _scalar_str(dd_train.get("inputDistribution"), "standard_normal")
    out_dist = _scalar_str(dd_train.get("outputDistribution"), "additive_gaussian")
    noise_level = _scalar_float(dd_train.get("noiseLevel"), 0.25)
    additive = out_dist == "additive_gaussian"
    sigma = noise_level if additive else 0.0
    input_dist_test = _scalar_str(dd_test.get("inputDistribution"), input_dist)
    out_dist_test = _scalar_str(dd_test.get("outputDistribution"), out_dist)
    noise_test = _scalar_float(dd_test.get("noiseLevel"), noise_level)
    additive_test = out_dist_test == "additive_gaussian"
    sigma_test = noise_test if additive_test else 0.0
    return _materialize_linear_like(
        ctx.rng, dd_train, dd_test, ctx.ds_test_raw, ctx.train_size, ctx.test_size,
        ctx.input_dim, ctx.output_dim,
        input_dist, sigma, additive, input_dist_test, sigma_test, additive_test,
    )


def preview_direct_arrays(req: "DatasetPreviewRequest"):
    """direct-arrays preview(engine 共享实现;effective_type 保持 per-type 分支)。"""
    from comfy_research.engine.datasets.dataset_preview_helpers import build_linear_or_symbolic_arrays

    return build_linear_or_symbolic_arrays(req.effective_type, req.data)
