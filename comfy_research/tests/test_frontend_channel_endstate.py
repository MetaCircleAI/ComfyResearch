"""Frontend definition and generated-source contracts.

node 的 schema/defaults/fields/family/hint/observable/resizable/adapter 键
全部单源 Python def;前端只消费生成物。手写残留 = 本文件枚举的 documented
exceptions(adapter 表/特例 spawn 分支/panel catalog),不再包含 schema facts。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FE = ROOT / "frontend"


def test_frontend_channel_holds_all_registered_defs() -> None:
    from comfy_research.nodes import registry
    from comfy_research.tests.test_frontend_channel_ledger import REGISTERED_FRONTEND_TYPES

    assert registry.frontend_def_types() == frozenset(REGISTERED_FRONTEND_TYPES)
    assert len(REGISTERED_FRONTEND_TYPES) == 51


def test_eight_registry_symmetry_holds() -> None:
    """八表对称查重存续(每个注册函数查其余全部注册表)。"""
    import inspect

    from comfy_research.nodes import registry

    tables = ("NODE_DEFS", "DATASET_DEFS", "MODEL_DEFS", "OPTIMIZER_DEFS",
              "LOSS_DEFS", "INITIALIZATION_DEFS", "TRAINER_DEFS", "FRONTEND_DEFS")
    fns = (registry.observable_def, registry.dataset_def, registry.model_def,
           registry.optimizer_def, registry.loss_def, registry.initialization_def,
           registry.trainer_def, registry.frontend_node_def)
    for fn in fns:
        src = inspect.getsource(fn)
        for table in tables:
            assert table in src, (fn.__name__, table)


def test_intentionally_hand_surfaces_enumerated() -> None:
    """终态手写残留 = declared adapter 面 + 特例分支,零 schema facts。"""
    # (a) 三张 adapter 表(组件/codegen/specCode):键→实现映射,throw-on-missing
    reg = (FE / "src" / "graph" / "nodeRegistry.ts").read_text()
    assert "GENERATED_COMPONENT_ADAPTERS" in reg and "has no such entry" in reg
    types_src = (FE / "src" / "graph" / "generatedNodeSpecTypes.ts").read_text()
    assert "CODEGEN_ADAPTERS" in types_src and "SPEC_CODE_ADAPTERS" in types_src
    # (b) ResearchCanvas 四特例 add-node 分支(options 线程,generic 排除名单)
    canvas = (FE / "src" / "components" / "ResearchCanvas.tsx").read_text()
    for t in ("observable_user", "linear_dataset", "symbolic_func_dataset", "combined_model"):
        assert f'nodeType !== "{t}"' in canvas, t
        assert f'if (nodeType === "{t}")' in canvas, t
    # (c) add 面板目录独立(panel catalog 不从 NODE_SPEC_REGISTRY 派生)
    modal = (FE / "src" / "components" / "AddNodeSearchModal.tsx").read_text()
    assert "useNodeCategories" in modal
    # (d) 生成器权威在 python(前端零 manifest 生产码)
    assert not list((FE / "scripts").glob("*node-manifest*"))
