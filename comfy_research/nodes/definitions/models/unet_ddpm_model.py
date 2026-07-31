"""unet_ddpm_model — compact image DDPM denoiser for CIFAR-10."""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, IntField, ModelDef


DEF = model_def(
    ModelDef(
        type="unet_ddpm_model",
        label="UNet DDPM",
        hint="Time-conditioned UNet noise predictor for CIFAR-10 image diffusion.",
        family=("diffusion_loss_model", "canvas_trainer_model_source", "canvas_full_model"),
        fields=(
            IntField(key="inChannels", label="Input Channels", default=3, min=1),
            IntField(key="baseChannels", label="Base Channels", default=64, min=8),
            EnumField(key="channelMult", label="Channel Multipliers", default="1,2,2"),
            IntField(key="timeEmbedDim", label="Time Embed Dim", default=128, min=8),
            IntField(key="diffusionTimesteps", label="Diffusion Timesteps", default=1000, min=2),
            IntField(key="imageSize", label="Image Size", default=32, min=8),
            IntField(key="seed", label="Seed", default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="UnetDdpmModelNode"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.unet_ddpm_model import build_unet_ddpm_from_md

    return build_unet_ddpm_from_md(data)
