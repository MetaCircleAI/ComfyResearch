"""toy-language 家族共享 provider 体。顶层 stdlib-only。

- **materialize 无 provider**:token 分支按
  `_TOY_LANGUAGE_TOKEN_DATASETS` capability 分派,ctx-override-from-shape 与
  text-heavy 256/64 回退是共享逻辑，避免复制到 13 个 provider。
- **preview**:builder 直调(dataset_tensor literal 分支原体);builder 表复用
  engine 的 `TOY_LM_BUILDERS`。
- **word-inspect 不归 provider**:dataset_tensor 的 post-dispatch 块按 capability
  再判,provider 只回 4 数组。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.nodes.provider_types import DatasetPreviewRequest


def preview_toy_language(req: "DatasetPreviewRequest"):
    """原 literal 分支体:builder(data, train_sz, test_sz),builder 查 TOY_LM_BUILDERS。"""
    from comfy_research.engine.datasets.dataset_runtime import TOY_LM_BUILDERS
    from comfy_research.engine.trainer.scalar import _scalar_int
    from comfy_research.schemas.graph import NodeKind

    data = req.data
    train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
    test_sz = max(0, _scalar_int(data.get("testSize"), 200))
    builder = TOY_LM_BUILDERS[NodeKind(req.effective_type)]
    return builder(data, train_sz, test_sz)
