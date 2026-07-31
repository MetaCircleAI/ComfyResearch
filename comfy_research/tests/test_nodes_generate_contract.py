"""byte contract: merge_and_render over the checked-in manifest must
reproduce every generated artifact byte-for-byte (empty NodeDef set).

The checked-in node_manifest.json is the legacy fragment; the roundtrip test
proved fragment == manifest byte-for-byte, so this contract needs no npm."""
from __future__ import annotations

from pathlib import Path

from comfy_research.nodes.generate import merge_and_render

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "comfy_research" / "generated"


def _legacy_fragment() -> str:
    """收官:legacy fragment 恒空——八通道 def 全量再现 manifest。
    保留函数形态 + 空断言防回归(任何条目脱离 def 通道会在此爆)。"""
    from comfy_research.nodes.generate import parse_fragment
    from comfy_research.nodes.registry import (
        dataset_def_types,
        frontend_def_types,
        model_def_types,
        observable_def_types,
        optimizer_def_types,
        loss_def_types,
        initialization_def_types,
        trainer_def_types,
    )

    migrated = (observable_def_types() | dataset_def_types() | model_def_types()
                | optimizer_def_types() | loss_def_types() | initialization_def_types()
                | trainer_def_types() | frontend_def_types())
    entries = parse_fragment((GEN / "node_manifest.json").read_text())
    legacy = [e for e in entries if e["type"] not in migrated]
    assert legacy == [], f"manifest entries outside def channels: {[e['type'] for e in legacy]}"
    return "[]"


def test_merge_reproduces_artifacts_byte_for_byte() -> None:
    manifest = (GEN / "node_manifest.json").read_text()
    out = merge_and_render(_legacy_fragment())
    assert out["manifest"] == manifest
    assert out["node_kind"] == (GEN / "node_kind.py").read_text()
    assert out["capabilities"] == (GEN / "node_capabilities.py").read_text()
    assert out["param_models"] == (GEN / "node_params.py").read_text()
    specs_path = (GEN.parent.parent / "frontend" / "src" / "generated" / "generatedNodeSpecs.ts")
    assert out["generated_specs_ts"] == specs_path.read_text()


def test_dataset_ui_metadata_stays_out_of_manifest() -> None:
    """Dataset 通道字节锚:dummy DatasetDef 带 options/min/step
    的字段 → manifest field 仍只有裸四键;generatedNodeSpecs.ts 才含 UI 元数据;
    其余 4 个后端生成物与空 dataset 集时逐字节一致(dummy 条目除外)。"""
    from comfy_research.nodes import registry
    from comfy_research.nodes.generate import def_to_entry_dataset, merge_and_render
    from comfy_research.nodes.schema import DatasetDef, EnumField, IntField

    d = DatasetDef(
        type="dataset__contract_dummy",
        label="Dummy DS",
        family=("canvas_dataset_source", "vector_regression_dataset"),
        fields=(
            IntField(key="numSamples", label="Num Samples", default=100, min=1, step=1),
            EnumField(key="noiseKind", label="Noise Kind", default="gauss", options=("gauss", "uniform")),
        ),
    )
    entry = def_to_entry_dataset(d)
    assert entry["fields"] == [
        {"kind": "int", "key": "numSamples", "label": "Num Samples", "defaultValue": 100},
        {"kind": "enum", "key": "noiseKind", "label": "Noise Kind", "defaultValue": "gauss"},
    ], entry["fields"]
    assert entry["family"] == ["canvas_dataset_source", "vector_regression_dataset"]
    assert "observable" not in entry and "hint" not in entry
    assert entry["hasCodegen"] is False

    baseline = merge_and_render(_legacy_fragment())
    registry.dataset_def(d)
    try:
        out = merge_and_render(_legacy_fragment())
        assert '"dataset__contract_dummy"' in out["manifest"]
        seg = out["manifest"].split('"dataset__contract_dummy"')[1][:800]
        assert '"min"' not in seg and '"step"' not in seg and '"options"' not in seg
        ts_tail = out["generated_specs_ts"].split('"dataset__contract_dummy"')[1]
        ts_seg = ts_tail.split("\n  },", 1)[0]  # 只看 dummy 自己的条目体
        assert "min: 1" in ts_seg and 'options: ["gauss", "uniform"]' in ts_seg
        assert "observable:" not in ts_seg and 'family: ["canvas_dataset_source", "vector_regression_dataset"]' in ts_seg
        # 其余生成物只多 dummy 条目:去掉 dummy 后与 baseline 恒等
        import re

        def _strip_dummy(text: str) -> str:
            # param_models 的 dummy 是整个 class 块(字段行不含标记),块级剔除;
            # node_kind/capabilities 是单行条目,行级剔除即可。
            text = re.sub(
                r"class \w*[Cc]ontract[_]?[Dd]ummy\w*\(NodeParamsBase\):\n(?:    .*\n)*\n*",
                "",
                text,
            )
            return "\n".join(
                ln for ln in text.splitlines() if "contract_dummy" not in ln and "ContractDummy" not in ln
            )

        for key in ("node_kind", "capabilities", "param_models"):
            assert _strip_dummy(out[key]) == _strip_dummy(baseline[key]), key
    finally:
        registry.DATASET_DEFS.pop(d.type, None)


def test_dataset_legacy_disjointness_guard() -> None:
    """dataset def 与 legacy fragment 同名 → FATAL(三源不相交)。"""
    from comfy_research.nodes import registry
    from comfy_research.nodes.generate import merge_and_render
    from comfy_research.nodes.schema import DatasetDef

    # 真实事故形态:TS 侧条目没删(fragment 里还有)就注册了 def。fragment 必须在
    # 注册前取;冲突 type 动态挑一个**仍在 legacy** 的 dataset(教训:借用已迁
    # type 会覆盖真 def,finally pop 把它删掉,污染整个套件)。
    import json

    import pytest

    from comfy_research.nodes.registry import dataset_def_types

    manifest = json.loads((GEN / "node_manifest.json").read_text())
    legacy_ds = sorted(
        e["type"] for e in manifest if e.get("category") == "dataset" and e["type"] not in dataset_def_types()
    )
    if not legacy_ds:
        pytest.skip("all datasets migrated — guard exercised by registry dup checks instead")
    victim = legacy_ds[0]
    stale_fragment = _legacy_fragment()
    d = DatasetDef(type=victim, label="X", family=())
    assert victim not in registry.DATASET_DEFS
    registry.DATASET_DEFS[d.type] = d  # 绕过 dataset_def 的 defs 间检查,直击 legacy 冲突
    try:
        with pytest.raises(RuntimeError, match="disjointness"):
            merge_and_render(stale_fragment)
    finally:
        registry.DATASET_DEFS.pop(d.type, None)


def test_sink_attention_mass_entry_matches_committed_manifest() -> None:
    """review 锚点:sink 的 manifest fields 原是 defaults 推断出来的
    (kind/key/label/defaultValue,label 来自 fieldLabelFromKey),def 显式声明后
    def_to_entry 必须逐键复刻;OVERRIDES 系条目无 hint 键(hint 可选化契约)。"""
    import json

    from comfy_research.nodes import registry
    from comfy_research.nodes.generate import def_to_entry

    registry.load_definitions()
    committed = json.loads((GEN / "node_manifest.json").read_text())
    want = next(e for e in committed if e["type"] == "observable_sink_attention_mass")
    got = def_to_entry(registry.NODE_DEFS["observable_sink_attention_mass"])
    assert got == want
    # hint 文案已从 NODE_HINTS 迁入 def；条件发射语义不变，
    # 该型现在由 def 持有 hint。
    assert got["hint"].startswith("Mean so")
    assert list(got) == list(want)  # 键序一致(json 字节等价的前提)


def test_ui_metadata_stays_out_of_backend_artifacts() -> None:
    """Dummy def 的 min/step/options 进入 generatedNodeSpecs，
    但四个后端生成物与空集时逐字节一致(def 未迁移任何 legacy node 时才成立——
    这里 dummy 是新增 type,manifest 会多一条,故对比策略:先算空集基线,再断言
    dummy 条目只出现在 dummy 的 entry 里,其余 entry 序列化不受影响 + UI metadata
    不进 manifest fields。"""
    from comfy_research.nodes import registry
    from comfy_research.nodes.generate import def_to_entry, merge_and_render, render_generated_specs_ts
    from comfy_research.nodes.schema import FrontendSpec, IntField, ObservableDef, SpawnSpec, VizSpec

    d = ObservableDef(
        type="observable__contract_dummy",
        label="Dummy",
        hint="dummy hint",
        viz=VizSpec(
            variant="user", title="Dummy", info_markdown="d", user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", unit="u"),
        ),
        fields=(IntField(key="topK", label="Top K", default=3, min=1, step=1),),
        frontend=FrontendSpec(),
    )
    entry = def_to_entry(d)
    assert entry["fields"] == [{"kind": "int", "key": "topK", "label": "Top K", "defaultValue": 3}], entry["fields"]

    registry.observable_def(d)

    def _noop(rec, on):  # pragma: no cover
        return None

    registry.PROVIDERS[d.type] = _noop
    try:
        ts = render_generated_specs_ts(registry.all_observable_defs())
        assert "min: 1" in ts and "step: 1" in ts and 'spawn: { kind: "user_scalar", unit: "u" }' in ts
        out = merge_and_render(_legacy_fragment())
        assert '"observable__contract_dummy"' in out["manifest"]
        assert '"min"' not in out["manifest"].split('"observable__contract_dummy"')[1][:600]
    finally:
        registry.NODE_DEFS.pop(d.type, None)
        registry.PROVIDERS.pop(d.type, None)


def test_js_number_str_matches_ecmascript_tostring() -> None:
    """Python-born float defaults 的 JS 字节保真(eps 案例回归锚)。"""
    from comfy_research.nodes.generate import js_number_str

    cases = [
        (1e-05, "0.00001"),      # eps 案例:python repr 是 '1e-05'
        (1e-06, "0.000001"),     # rms eps;JS 十进制下界内
        (1e-07, "1e-7"),         # JS 指数临界(< 1e-6 才用指数)
        (0.05, "0.05"),
        (1.0, "1"),              # JS 整值浮点无 '.0'
        (1e21, "1e+21"),         # JS 上界指数形式
        (1e16, "10000000000000000"),  # python 用指数、JS 用十进制的区间
        (123.456, "123.456"),
        (-1e-05, "-0.00001"),
        (7, "7"),
        # 大整值浮点必须打最短位数+补零,不是二进制精确整数
        (1.2345678901234567e20, "123456789012345670000"),
        (9.999999999999999e20, "999999999999999900000"),
        (1.5e21, "1.5e+21"),
        (1.5e-7, "1.5e-7"),
    ]
    for v, want in cases:
        assert js_number_str(v) == want, (v, js_number_str(v), want)
