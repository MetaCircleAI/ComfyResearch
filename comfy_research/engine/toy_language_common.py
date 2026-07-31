"""Permanent notebook/codegen compatibility shim.

前端 codegen 生成的 notebook 代码 import 本旧路径(白名单见
tests/test_notebook_contract_modules.NOTEBOOK_CONTRACT_SHIMS)。真身在
comfy_research/engine/datasets/toy_language_common.py;显式 re-export 公有面。
repo 内部代码禁止 import 此路径。
"""
from __future__ import annotations

from comfy_research.engine.datasets.toy_language_common import (
    dataset_rng_seed,
    resize_sequence,
    scalar_float,
    scalar_int,
    scalar_str,
    slice_last_token_lm,
    slice_shifted_window_lm,
)

__all__ = [
    "dataset_rng_seed",
    "resize_sequence",
    "scalar_float",
    "scalar_int",
    "scalar_str",
    "slice_last_token_lm",
    "slice_shifted_window_lm",
]
