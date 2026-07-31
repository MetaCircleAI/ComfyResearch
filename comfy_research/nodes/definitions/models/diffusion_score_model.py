"""diffusion_score_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest。六个 int 字段生成 sweep 轴，不生成 enum 轴。
builder 优先使用 context.input_dim，并回退到 metadata。
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="diffusion_score_model",
        label='Diffusion score MLP',
        hint="MLP εθ(x_t, t) for DDPM-style noise prediction on vectors; use with diffusion_mse_loss + vector datasets.",
        family=('diffusion_loss_model', 'canvas_trainer_model_source', 'canvas_full_model'),
        fields=(
            IntField(key="inputDim", label='Input Dim', default=8, min=1),
            IntField(key="hiddenDim", label='Hidden Dim', default=128, min=8),
            IntField(key="depth", label='Depth', default=3, min=1),
            IntField(key="timeEmbedDim", label='Time Embed Dim', default=64, min=8),
            IntField(key="diffusionTimesteps", label='Diffusion Timesteps', default=100, min=2),
            IntField(key="seed", label='Seed', default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="DiffusionScoreModelNode", codegen_key="diffusion_score_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_diffusion_score_model

    return _build_diffusion_score_model(data, context)
