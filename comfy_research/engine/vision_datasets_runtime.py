"""Permanent notebook/codegen compatibility shim.

前端 codegen 生成的 notebook 代码 import 本旧路径(白名单见
tests/test_notebook_contract_modules.NOTEBOOK_CONTRACT_SHIMS)。真身在
comfy_research/engine/datasets/vision_datasets_runtime.py;显式 re-export 公有面。
repo 内部代码禁止 import 此路径。
"""
from __future__ import annotations

from comfy_research.engine.datasets.vision_datasets_runtime import (
    build_cifar10_arrays,
    build_gaussian_blob_arrays,
    build_hole_counting_arrays,
    build_mnist_arrays,
    build_shape_world_arrays,
    build_vision_numpy_arrays,
    vision_num_classes,
)

__all__ = [
    "build_cifar10_arrays",
    "build_gaussian_blob_arrays",
    "build_hole_counting_arrays",
    "build_mnist_arrays",
    "build_shape_world_arrays",
    "build_vision_numpy_arrays",
    "vision_num_classes",
]
