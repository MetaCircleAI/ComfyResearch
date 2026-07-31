"""Permanent notebook/codegen compatibility shim.

前端 codegen 生成的 notebook 代码 import 本旧路径(白名单见
tests/test_notebook_contract_modules.NOTEBOOK_CONTRACT_SHIMS)。真身在
comfy_research/engine/datasets/random_input_distribution_runtime.py;显式 re-export 公有面。
repo 内部代码禁止 import 此路径。
"""
from __future__ import annotations

from comfy_research.engine.datasets.random_input_distribution_runtime import (
    rng_from_random_input_distribution_data,
    sample_inputs,
    sample_x_from_random_input_dict,
    sample_x_from_sampler_dict,
    scalar_float,
    scalar_int,
    scalar_str,
)

__all__ = [
    "rng_from_random_input_distribution_data",
    "sample_inputs",
    "sample_x_from_random_input_dict",
    "sample_x_from_sampler_dict",
    "scalar_float",
    "scalar_int",
    "scalar_str",
]
