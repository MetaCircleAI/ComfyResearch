"""Permanent notebook/codegen compatibility shim.

前端 optimizerCodegen 生成的 notebook 代码 import 本旧路径(Shampoo/SOAP;
白名单见 tests/test_notebook_contract_modules.NOTEBOOK_CONTRACT_SHIMS)。
真身在 comfy_research/engine/optimizers/matrix_preconditioner_optimizers.py。
repo 内部代码禁止 import 此路径。
"""
from __future__ import annotations

from comfy_research.engine.optimizers.matrix_preconditioner_optimizers import SOAP, Shampoo

__all__ = ["Shampoo", "SOAP"]
