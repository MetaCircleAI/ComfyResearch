"""Notebook-contract module。

前端 codegen 为 symbolic_func_dataset 生成的 notebook cell 一直发射
``from comfy_research.engine.linear_dataset_sampling import sample_inputs``,
但本模块历史上从未存在——该 cell 对用户一直是断的(既有 bug)。真身在
random_input_distribution_runtime;这里显式 re-export,同时修复历史导出
与未来导出的 notebook,codegen 字符串与 codegen golden 不动。

This module preserves the notebook/codegen import contract. Internal Python
code should import the implementation from ``engine.datasets`` directly.
"""
from __future__ import annotations

from comfy_research.engine.datasets.random_input_distribution_runtime import sample_inputs

__all__ = ["sample_inputs"]
