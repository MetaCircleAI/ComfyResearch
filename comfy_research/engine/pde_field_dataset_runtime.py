"""Permanent notebook/codegen compatibility shim.

前端 codegen 生成的 notebook 代码 import 本旧路径(白名单见
tests/test_notebook_contract_modules.NOTEBOOK_CONTRACT_SHIMS)。真身在
comfy_research/engine/datasets/pde_field_dataset_runtime.py;显式 re-export 公有面。
repo 内部代码禁止 import 此路径。
"""
from __future__ import annotations

from comfy_research.engine.datasets.pde_field_dataset_runtime import (
    build_pde_field_arrays,
    pde_field_flat_dims,
)

__all__ = [
    "build_pde_field_arrays",
    "pde_field_flat_dims",
]
