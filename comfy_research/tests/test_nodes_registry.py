"""comfy_research.nodes registry unit tests + import-cycle AST guard."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from comfy_research.nodes import registry
from comfy_research.nodes.schema import DatasetDef, FrontendSpec, ObservableDef, VizSpec

NODES_DIR = Path(__file__).resolve().parents[1] / "nodes"


def _dummy(type_: str) -> ObservableDef:
    return ObservableDef(
        type=type_,
        label="Dummy",
        hint="dummy",
        viz=VizSpec(variant="user", title="Dummy", info_markdown="d", user_whitelisted=True),
        frontend=FrontendSpec(),
    )


def test_channel_is_valid_and_sorted() -> None:
    defs = registry.all_observable_defs()
    types = [d.type for d in defs]
    assert types == sorted(types)
    registry.validate_defs()


def test_duplicate_registration_raises() -> None:
    d = _dummy("observable__test_dup")
    registry.observable_def(d)
    try:
        with pytest.raises(RuntimeError, match="duplicate NodeDef"):
            registry.observable_def(d)
    finally:
        registry.NODE_DEFS.pop("observable__test_dup", None)


def test_def_without_recorder_fails_validation() -> None:
    d = _dummy("observable__test_orphan")
    registry.observable_def(d)
    try:
        with pytest.raises(RuntimeError, match="without recorder"):
            registry.validate_defs()
    finally:
        registry.NODE_DEFS.pop("observable__test_orphan", None)


def _dummy_dataset(type_: str) -> DatasetDef:
    return DatasetDef(type=type_, label="Dummy DS", family=("canvas_dataset_source",))


def test_dataset_channel_is_valid_and_sorted() -> None:
    defs = registry.all_dataset_defs()
    types = [d.type for d in defs]
    assert types == sorted(types)
    registry.validate_defs()


def test_dataset_duplicate_registration_raises() -> None:
    d = _dummy_dataset("dataset__test_dup")
    registry.dataset_def(d)
    try:
        with pytest.raises(RuntimeError, match="duplicate NodeDef"):
            registry.dataset_def(d)
        # 跨通道同名也必须 FATAL(dataset_def 检查 NODE_DEFS)。
        with pytest.raises(RuntimeError, match="duplicate NodeDef"):
            registry.dataset_def(DatasetDef(type="observable_weight_l2", label="X", family=()))
    finally:
        registry.DATASET_DEFS.pop("dataset__test_dup", None)


def test_dataset_orphan_preview_fails_validation() -> None:
    """role-based 验证:provider 槽位可选,但孤儿 preview FATAL;
    def 无任何 provider 是合法的(auxiliary node 没有 materializer)。"""
    d = _dummy_dataset("dataset__test_no_provider")
    registry.dataset_def(d)
    try:
        registry.validate_defs()  # builder/preview 缺席合法
    finally:
        registry.DATASET_DEFS.pop("dataset__test_no_provider", None)

    registry.DATASET_PREVIEWS["dataset__test_orphan_preview"] = lambda req: None
    try:
        with pytest.raises(RuntimeError, match="preview provider"):
            registry.validate_defs()
    finally:
        registry.DATASET_PREVIEWS.pop("dataset__test_orphan_preview", None)


def test_nodes_package_never_toplevel_imports_heavy_or_recorder() -> None:
    """import-cycle 硬规(spec §2):nodes 包顶层不得 import torch/numpy/recorder。"""
    banned = ("torch", "numpy", "comfy_research.engine.trainer.recorder")
    for path in NODES_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:  # 只查模块顶层;TYPE_CHECKING 的 If 分支不在 tree.body 的 import 之列
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert not any(n == b or n.startswith(b + ".") for b in banned), (path.name, n)
