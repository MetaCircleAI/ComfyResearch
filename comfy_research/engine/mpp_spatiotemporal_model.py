"""Permanent notebook/codegen compatibility shim.

前端 codegen 生成的 notebook 代码 import 本旧路径(白名单见
tests/test_notebook_contract_modules.NOTEBOOK_CONTRACT_SHIMS)。真身在
comfy_research/engine/models/mpp_spatiotemporal_model.py;显式 re-export 公有面
Exports are explicit: no wildcard import and no ``__all__`` change in the implementation module.
repo 内部代码禁止 import 此路径。
"""
from __future__ import annotations

from comfy_research.engine.models.mpp_spatiotemporal_model import (
    MppSpatiotemporalModel,
    mpp_spatiotemporal_from_canvas_md,
)

__all__ = [
    "MppSpatiotemporalModel",
    "mpp_spatiotemporal_from_canvas_md",
]
