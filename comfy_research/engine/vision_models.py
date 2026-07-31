"""Permanent notebook/codegen compatibility shim.

前端 codegen 生成的 notebook 代码 import 本旧路径(白名单见
tests/test_notebook_contract_modules.NOTEBOOK_CONTRACT_SHIMS)。真身在
comfy_research/engine/models/vision_models.py;显式 re-export 公有面
。
repo 内部代码禁止 import 此路径。
"""
from __future__ import annotations

from comfy_research.engine.models.vision_models import (
    SmallResNet,
    TinyViT,
    build_resnet_from_md,
    build_vit_from_md,
    infer_vision_input_channels_height_width,
)

__all__ = [
    "SmallResNet",
    "TinyViT",
    "build_resnet_from_md",
    "build_vit_from_md",
    "infer_vision_input_channels_height_width",
]
