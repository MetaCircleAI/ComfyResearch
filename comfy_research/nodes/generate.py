"""Merge generator: legacy TS fragment + Python NodeDefs -> generated artifacts.

Byte-fidelity strategy: the fragment is parsed with lexeme-preserving numbers
(JS renders 1e-6 as ``0.000001`` and 1e-8 as ``1e-8``; Python's float repr
differs), and re-emitted verbatim by a JS-style JSON writer. The Python
serializers below replicate frontend/scripts/nodeManifestCore.ts character for
character — the contract test (test_nodes_generate_contract.py) locks this.

stdlib-only and runnable as a plain file (sys.path bootstrap), so the frontend
wrapper needs no conda env.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfy_research.nodes import registry  # noqa: E402
from comfy_research.nodes.schema import ObservableDef  # noqa: E402

GEN_DIR = ROOT / "comfy_research" / "generated"
SPECS_TS_PATH = ROOT / "frontend" / "src" / "generated" / "generatedNodeSpecs.ts"

OUT_FILES = {
    "manifest": GEN_DIR / "node_manifest.json",
    "node_kind": GEN_DIR / "node_kind.py",
    "capabilities": GEN_DIR / "node_capabilities.py",
    "param_models": GEN_DIR / "node_params.py",
    "generated_specs_ts": SPECS_TS_PATH,
}


class RawNum(str):
    """JSON number carrying its exact source lexeme."""


def parse_fragment(text: str) -> list:
    return json.loads(text, parse_float=RawNum, parse_int=RawNum, parse_constant=RawNum)



def js_number_str(v) -> str:
    """ECMAScript Number::toString for Python-born defaults(byte parity with JS
    JSON.stringify 的 eps 案例：Python json.dumps(1e-05)='1e-05'，JS 是
    '0.00001')。数字位一律取自最短 roundtrip repr；大整值浮点不可用
    int(f)——那是二进制精确值,JS 打的是最短位数 + 补零);阈值:首位十进制指数
    ∈ (-7, 21) 用定点,否则指数形式。"""
    if isinstance(v, bool):
        raise TypeError("bool is not a number")
    if isinstance(v, int):
        return str(v)
    from decimal import Decimal

    d = Decimal(repr(float(v)))
    sign, digits, exp = d.as_tuple()
    msd_exp = exp + len(digits) - 1
    if -7 < msd_exp < 21:
        s = format(d, "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    mant_digits = "".join(map(str, digits)).rstrip("0") or "0"
    mant = mant_digits[0] + (f".{mant_digits[1:]}" if len(mant_digits) > 1 else "")
    return f"{'-' if sign else ''}{mant}e{'+' if msd_exp >= 0 else '-'}{abs(msd_exp)}"


def emit_json(value, indent: int = 0) -> str:
    """JS-style JSON.stringify(value, null, 2) with lexeme-preserving numbers."""
    pad, pad_in = "  " * indent, "  " * (indent + 1)
    if isinstance(value, RawNum):
        return str(value)
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):  # Python-born numbers (NodeDef defaults)
        return js_number_str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        items = ",\n".join(f"{pad_in}{emit_json(v, indent + 1)}" for v in value)
        return f"[\n{items}\n{pad}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = ",\n".join(
            f"{pad_in}{json.dumps(k, ensure_ascii=False)}: {emit_json(v, indent + 1)}" for k, v in value.items()
        )
        return f"{{\n{items}\n{pad}}}"
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def emit_json_compact(value) -> str:
    """JS-style JSON.stringify(value) (no whitespace), lexeme-preserving."""
    if isinstance(value, RawNum):
        return str(value)
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return js_number_str(value)
    if isinstance(value, list):
        return "[" + ",".join(emit_json_compact(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{json.dumps(k, ensure_ascii=False)}:{emit_json_compact(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


# ---- serializers: character-for-character ports of nodeManifestCore.ts ----

def serialize_manifest(entries: list) -> str:
    return emit_json(entries) + "\n"


def serialize_node_kind(entries: list) -> str:
    lines = [f"    {e['type']} = {json.dumps(e['type'])}" for e in entries]
    joined = "\n".join(lines)
    return f"""from __future__ import annotations

from enum import Enum


# Generated from frontend/src/graph/nodeRegistrySpec.ts.
# Do not edit by hand; run `npm run generate:node-manifest` from frontend/.
class GeneratedNodeKind(str, Enum):
{joined}


GENERATED_NODE_KIND_VALUES: tuple[str, ...] = tuple(kind.value for kind in GeneratedNodeKind)
"""


def serialize_capabilities(entries: list) -> str:
    lines = [
        f"    {json.dumps(e['type'])}: frozenset({emit_json_compact(list(e['family']))}),"
        for e in entries
        if e.get("family")
    ]
    joined = "\n".join(lines)
    return f"""from __future__ import annotations

from collections.abc import Iterable


# Generated from frontend/src/graph/nodeRegistrySpec.ts.
# Do not edit by hand; run `npm run generate:node-manifest` from frontend/.
NODE_CAPABILITIES: dict[str, frozenset[str]] = {{
{joined}
}}


def _node_type_key(node_type: object) -> str:
    return str(getattr(node_type, "value", node_type))


def capabilities_for(node_type: object) -> frozenset[str]:
    return NODE_CAPABILITIES.get(_node_type_key(node_type), frozenset())


def has_capability(node_type: object, capability: str) -> bool:
    return str(capability) in capabilities_for(node_type)


def node_types_with_capability(capability: str) -> tuple[str, ...]:
    cap = str(capability)
    return tuple(sorted(node_type for node_type, caps in NODE_CAPABILITIES.items() if cap in caps))


def has_all_capabilities(node_type: str, capabilities: Iterable[str]) -> bool:
    caps = capabilities_for(str(node_type))
    return all(str(capability) in caps for capability in capabilities)
"""


def class_name_for_node_type(type_: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in type_.split("_") if part) + "Params"


def py_literal(value) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value is None:
        return "None"
    return emit_json_compact(value)


def python_type_for_field(field: dict) -> str:
    kind = field["kind"]
    if kind in ("int", "intList"):
        return "int | list[int]"
    if kind in ("float", "floatList"):
        return "float | list[float]"
    if kind == "boolean":
        return "bool | list[bool]"
    if kind in ("enum", "string"):
        return "str | list[str]"
    raise ValueError(f"unknown field kind: {kind!r}")


def serialize_param_models(entries: list) -> str:
    classes = []
    for e in entries:
        fields = e.get("fields") or []
        cls = class_name_for_node_type(e["type"])
        if not fields:
            classes.append(f"class {cls}(NodeParamsBase):\n    pass\n")
            continue
        lines = "\n".join(
            f"    {f['key']}: {python_type_for_field(f)} = {py_literal(f['defaultValue'])}" for f in fields
        )
        classes.append(f"class {cls}(NodeParamsBase):\n{lines}\n")
    mappings = "\n".join(f"    {json.dumps(e['type'])}: {class_name_for_node_type(e['type'])}," for e in entries)
    joined_classes = "\n".join(classes)
    return f"""from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


# Generated from frontend/src/graph/nodeRegistrySpec.ts.
# Do not edit by hand; run `npm run generate:node-manifest` from frontend/.
class NodeParamsBase(BaseModel):
    model_config = ConfigDict(extra="allow")


{joined_classes}
NODE_PARAM_MODELS: dict[str, type[NodeParamsBase]] = {{
{mappings}
}}


def param_model_for(node_type: str) -> type[NodeParamsBase]:
    return NODE_PARAM_MODELS.get(str(node_type), NodeParamsBase)


def validate_node_params(node_type: str, data: dict[str, Any] | None) -> NodeParamsBase:
    return param_model_for(str(node_type)).model_validate(data or {{}})
"""


# ---- NodeDef channel ----

def def_to_entry(def_: ObservableDef) -> dict:
    """Render an ObservableDef into a ManifestEntry-shaped dict (buildManifest key order)."""
    fields = []
    for f in def_.fields:
        entry: dict = {"kind": f.kind, "key": f.key, "label": f.label, "defaultValue": f.default}
        # options 入 manifest 由 manifest_options 旋钮控制(kan_reg 的
        # regMetric manifest 裸、generatedNodeSpecs 带 options;既有 observable
        # defs 无 options,零字节变化)。
        options = getattr(f, "options", None)
        if options and getattr(f, "manifest_options", False):
            entry["options"] = list(options)
        fields.append(entry)
    out: dict = {
        "type": def_.type,
        "label": def_.label,
        "category": def_.category,
        # hint 条件发射,truthy 语义镜像 buildManifest(`spec.hint ? {hint} : {}`):
        # OVERRIDES 系条目没有 hint 键;键序保持 category → hint → family。
        **({"hint": def_.hint} if def_.hint else {}),
        "family": ["observable"],
        "observable": {
            "infoMarkdown": def_.viz.info_markdown,
            "vizTitle": def_.viz.title,
            "vizVariant": def_.viz.variant,
            "spawnsVizNode": def_.viz.spawns,
        },
        "resizable": True,
        "hasDefaults": True,
        "hasCodegen": def_.frontend.codegen_key is not None,
    }
    if fields:
        out["fields"] = fields
    return out


GENERATED_SPECS_HEADER = (
    "// Generated by comfy_research/nodes/generate.py. Do not edit by hand;\n"
    "// run `npm run generate:node-manifest` from frontend/.\n"
)


def _ts_str(v: str) -> str:
    return json.dumps(v, ensure_ascii=False)


def _ts_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, str):
        return _ts_str(v)
    if isinstance(v, (int, float)):
        return js_number_str(v)
    if isinstance(v, (list, tuple)):  # spawn_defaults 的 shape 列表
        return "[" + ", ".join(_ts_value(x) for x in v) + "]"
    if isinstance(v, dict):  # 嵌套/空对象 defaults（activationTensorCaches: {}）
        if not v:
            return "{}"
        return "{ " + ", ".join(f"{_ts_str(k)}: {_ts_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"unsupported TS scalar: {type(v)!r}")


def _render_ports_ts(ports) -> str | None:
    """ports 块只进 generatedNodeSpecs(manifest 零涉及,options/min 同律)。"""
    if not ports:
        return None
    items = []
    for p in ports:
        accepts = []
        for a in p.accepts:
            parts = []
            if a.source_family is not None:
                parts.append(f"family: {_ts_str(a.source_family)}")
            if a.source_type is not None:
                parts.append(f"type: {_ts_str(a.source_type)}")
            if getattr(a, "source_io_mode", None) is not None:
                # ioMode 条件接受(readNodeCanvasIoMode 镜像)。
                parts.append(f"ioMode: {_ts_str(a.source_io_mode)}")
            parts.append("handles: [" + ", ".join(_ts_str(h) for h in a.handles) + "]")
            accepts.append("{ " + ", ".join(parts) + " }")
        items.append("{ id: " + _ts_str(p.id) + ", accepts: [" + ", ".join(accepts) + "] }")
    return "    ports: { in: [" + ", ".join(items) + "] },"


def _render_field_ts(f) -> str:
    parts = [f"kind: {_ts_str(f.kind)}", f"key: {_ts_str(f.key)}", f"label: {_ts_str(f.label)}",
             f"defaultValue: {_ts_value(f.default)}"]
    # UI-only metadata:min/max/step/options 只进 generatedNodeSpecs,不进 manifest
    for attr in ("min", "max", "step"):
        val = getattr(f, attr, None)
        if val is not None:
            parts.append(f"{attr}: {_ts_value(val)}")
    options = getattr(f, "options", None)
    if options:
        parts.append("options: [" + ", ".join(_ts_str(o) for o in options) + "]")
    if getattr(f, "positive_only", False):
        parts.append("positiveOnly: true")
    tooltip = getattr(f, "tooltip", None)
    if tooltip:
        parts.append(f"tooltip: {_ts_str(tooltip)}")
    aria = getattr(f, "aria_label", None)
    if aria:
        parts.append(f"ariaLabel: {_ts_str(aria)}")
    if getattr(f, "sweepable", True) is False:
        parts.append("sweepable: false")
    if getattr(f, "sweep_kind", None) is not None:
        parts.append(f'sweepKind: {_ts_str(f.sweep_kind)}')
    return "{ " + ", ".join(parts) + " }"


def _render_def_ts(d: ObservableDef) -> str:
    lines = [f"  {_ts_str(d.type)}: {{"]
    lines.append(f"    label: {_ts_str(d.label)},")
    lines.append(f"    category: {_ts_str(d.category)},")
    if d.hint:
        # generated specs 同样不发 key(绝不发 hint: "")——两份 truth 各自条件发射。
        lines.append(f"    hint: {_ts_str(d.hint)},")
    if d.category != "observables":
        # specFromGenerated 只在 category==observables 时推断
        # family ["observable"]——kan_reg(category=loss)需显式发射,否则前端
        # 注册表丢 observable capability(verify-node-registry 契约)。
        lines.append('    family: ["observable"],')
    defaults_items = ", ".join(f"{_ts_str(f.key)}: {_ts_value(f.default)}" for f in d.fields)
    lines.append(f"    defaults: {{ {defaults_items} }}," if d.fields else "    defaults: {},")
    if d.fields:
        lines.append("    fields: [" + ", ".join(_render_field_ts(f) for f in d.fields) + "],")
    ports_line = _render_ports_ts(getattr(d, "ports", None))
    if ports_line:
        lines.append(ports_line)
    lines.append("    observable: {")
    lines.append(f"      infoMarkdown: {_ts_str(d.viz.info_markdown)},")
    lines.append(f"      vizTitle: {_ts_str(d.viz.title)},")
    lines.append(f"      vizVariant: {_ts_str(d.viz.variant)},")
    lines.append(f"      spawnsVizNode: {_ts_value(d.viz.spawns)},")
    lines.append("    },")
    spawn = getattr(d.viz, "spawn", None)
    if spawn is not None:
        sp = [f"kind: {_ts_str(spawn.kind)}"]
        if spawn.title is not None:
            sp.append(f"title: {_ts_str(spawn.title)}")
        if spawn.fixed_top_k is not None:
            sp.append(f"fixedTopK: {_ts_value(spawn.fixed_top_k)}")
        if spawn.top_k_from_field is not None:
            sp.append(f"topKFromField: {_ts_str(spawn.top_k_from_field)}")
        if spawn.unit is not None:
            sp.append(f"unit: {_ts_str(spawn.unit)}")
        if spawn.order_from_field is not None:
            sp.append(f"orderFromField: {_ts_str(spawn.order_from_field)}")
        if spawn.series_labels is not None:
            sp.append("seriesLabels: [" + ", ".join(_ts_str(s) for s in spawn.series_labels) + "]")
        if spawn.title_from_field is not None:
            sp.append(f"titleFromField: {_ts_str(spawn.title_from_field)}")
        lines.append("    spawn: { " + ", ".join(sp) + " },")
    fr = ["componentKey: " + _ts_str(d.frontend.component_key)]
    if d.frontend.codegen_key is not None:
        fr.append("codegenKey: " + _ts_str(d.frontend.codegen_key))
    lines.append("    frontend: { " + ", ".join(fr) + " },")
    lines.append("  },")
    return "\n".join(lines)


def render_generated_specs_ts(defs: tuple[ObservableDef, ...]) -> str:
    head = GENERATED_SPECS_HEADER + (
        'import type { GeneratedNodeSpec } from "../graph/generatedNodeSpecTypes";\n\n'
    )
    if not defs:
        return head + "export const GENERATED_NODE_SPECS: Record<string, GeneratedNodeSpec> = {};\n"
    body = "\n".join(_render_def_ts(d) for d in sorted(defs, key=lambda d: d.type))
    return head + "export const GENERATED_NODE_SPECS: Record<string, GeneratedNodeSpec> = {\n" + body + "\n};\n"


def merge_entries(legacy: list, defs: tuple[ObservableDef, ...]) -> list:
    if not defs:
        return legacy
    legacy_types = {e["type"] for e in legacy}
    def_types = {d.type for d in defs}
    overlap = legacy_types & def_types
    if overlap:
        raise RuntimeError(f"node type(s) defined in BOTH channels (disjointness violated): {sorted(overlap)}")
    merged = list(legacy) + [def_to_entry(d) for d in defs]
    merged.sort(key=lambda e: e["type"])
    return merged


def def_to_entry_dataset(def_) -> dict:
    """DatasetDef → ManifestEntry-shaped dict(dataset 通道字节锚)。

    与 def_to_entry(observable)刻意分离:dataset 的 legacy manifest
    字段恒为裸四键 {kind,key,label,defaultValue}——options/min/max/step 是
    UI+sweep truth,只进 generatedNodeSpecs.ts,绝不进 manifest(observable 通道
    的 options 会进 manifest;两通道各自与自己的 legacy 字节形态对齐)。
    """
    fields = [
        {"kind": f.kind, "key": f.key, "label": f.label, "defaultValue": f.default}
        for f in def_.fields
    ]
    out: dict = {
        "type": def_.type,
        "label": def_.label,
        "category": def_.category,
        **({"hint": def_.hint} if def_.hint else {}),
        # family 条件发射:None → manifest 无该键(auxiliary node)。
        **({"family": list(def_.family)} if def_.family is not None else {}),
        "resizable": True,
        "hasDefaults": True,
        "hasCodegen": def_.frontend.codegen_key is not None,
    }
    if fields:
        out["fields"] = fields
    return out


def def_to_entry_model(def_) -> dict:
    """ModelDef → ManifestEntry-shaped dict(model 通道字节锚;dataset 同款纪律:
    fields 裸四键,options/min/max/sweepable 只进 generatedNodeSpecs)。
    hasCodegen 派生自 codegen_key;hasDefaults 需按 manifest 实况(combined_model/
    tensor_add 为 false)——由 has_defaults 显式传递?模型 manifest 有 2 个
    hasDefaults:false → ModelDef 无该位,按 fields 非空或 defaults 恒 true 不成立,
    故这里按 def.fields 判断不可靠——采用显式规则:hasDefaults = bool(fields) or
    type not in _MODEL_NO_DEFAULTS。"""
    fields = [
        {"kind": f.kind, "key": f.key, "label": f.label, "defaultValue": f.default}
        for f in def_.fields
    ]
    out: dict = {
        "type": def_.type,
        "label": def_.label,
        "category": def_.category,
        **({"hint": def_.hint} if def_.hint else {}),
        **({"family": list(def_.family)} if def_.family is not None else {}),
        "resizable": True,
        "hasDefaults": def_.type not in _MODEL_NO_DEFAULTS,
        "hasCodegen": def_.frontend.codegen_key is not None,
    }
    if fields:
        out["fields"] = fields
    return out


# Only combined_model omits defaults; tensor_add includes runtime defaults for
# outputTensor and lastError.
_MODEL_NO_DEFAULTS = frozenset({"combined_model"})


def def_to_entry_optimizer(def_) -> dict:
    """OptimizerDef → ManifestEntry dict。fields 四键裸，富 UI 元数据/ui 块/
    specCode key 只进 generatedNodeSpecs)。9 型 manifest 实况:hasDefaults/
    hasCodegen 恒 true、resizable 恒 true。"""
    fields = [
        {"kind": f.kind, "key": f.key, "label": f.label, "defaultValue": f.default}
        for f in def_.fields
    ]
    out: dict = {
        "type": def_.type,
        "label": def_.label,
        "category": def_.category,
        **({"hint": def_.hint} if def_.hint else {}),
        **({"family": list(def_.family)} if def_.family is not None else {}),
        "resizable": True,
        "hasDefaults": True,
        "hasCodegen": def_.frontend.codegen_key is not None,
    }
    if fields:
        out["fields"] = fields
    return out


def def_to_entry_loss(def_) -> dict:
    """LossDef → ManifestEntry dict。hasCodegen 派生 codegen_key(l1/l2/
    l2_projection 无 codegen 为 manifest 实况);manifest_options 旋钮控制 enum
    options 入 manifest(mse 的 spec-level fields 实况)。"""
    fields = [
        {"kind": f.kind, "key": f.key, "label": f.label, "defaultValue": f.default,
         **({"options": list(f.options)} if getattr(f, "manifest_options", False) else {})}
        for f in def_.fields
    ]
    out: dict = {
        "type": def_.type,
        "label": def_.label,
        "category": def_.category,
        **({"hint": def_.hint} if def_.hint else {}),
        **({"family": list(def_.family)} if def_.family is not None else {}),
        "resizable": True,
        "hasDefaults": True,
        "hasCodegen": def_.frontend.codegen_key is not None,
    }
    if fields:
        out["fields"] = fields
    return out


def def_to_entry_initialization(def_) -> dict:
    """InitializationDef → ManifestEntry dict(I1;三型实况 hasDefaults/hasCodegen
    恒 true、resizable 恒 true、无 family;saxe/tau fields 裸)。"""
    fields = [
        {"kind": f.kind, "key": f.key, "label": f.label, "defaultValue": f.default}
        for f in def_.fields
    ]
    out: dict = {
        "type": def_.type,
        "label": def_.label,
        "category": def_.category,
        **({"hint": def_.hint} if def_.hint else {}),
        **({"family": list(def_.family)} if def_.family is not None else {}),
        "resizable": True,
        "hasDefaults": True,
        "hasCodegen": def_.frontend.codegen_key is not None,
    }
    if fields:
        out["fields"] = fields
    return out


def def_to_entry_trainer(def_) -> dict:
    """TrainerDef → ManifestEntry dict。spec-level fields 的 computeDevice
    options 入 manifest = manifest_options 旋钮)。"""
    fields = [
        {"kind": f.kind, "key": f.key, "label": f.label, "defaultValue": f.default,
         **({"options": list(f.options)} if getattr(f, "manifest_options", False) else {})}
        for f in def_.fields
    ]
    out: dict = {
        "type": def_.type,
        "label": def_.label,
        "category": def_.category,
        **({"hint": def_.hint} if def_.hint else {}),
        **({"family": list(def_.family)} if def_.family is not None else {}),
        "resizable": True,
        "hasDefaults": True,
        "hasCodegen": def_.frontend.codegen_key is not None,
    }
    if fields:
        out["fields"] = fields
    return out


def def_to_entry_frontend(def_) -> dict:
    """FrontendNodeDef → ManifestEntry dict(第八通道字节锚)。
    键序与 nodeManifestCore.buildManifest 恒等:type,label,category,hint?,family?,
    observable?,resizable,hasDefaults,hasCodegen,fields?。observable 子键按
    buildManifest 的 truthy 语义条件发射(spawnsVizNode 用 != null)。
    hasDefaults = defaults is not None(三态:None/()/非空)。"""
    fields = [
        {"kind": f.kind, "key": f.key, "label": f.label, "defaultValue": f.default}
        for f in def_.fields
    ]
    observable = {}
    if def_.observable is not None:
        ob = def_.observable
        if ob.info_markdown:
            observable["infoMarkdown"] = ob.info_markdown
        if ob.viz_title:
            observable["vizTitle"] = ob.viz_title
        if ob.viz_variant:
            observable["vizVariant"] = ob.viz_variant
        if ob.spawns_viz_node is not None:
            observable["spawnsVizNode"] = ob.spawns_viz_node
    out: dict = {
        "type": def_.type,
        "label": def_.label,
        "category": def_.category,
        **({"hint": def_.hint} if def_.hint else {}),
        **({"family": list(def_.family)} if def_.family is not None else {}),
        # 门控与 buildManifest 恒等同构:spec.observable truthy 即发块
        # (子键内部再 truthy 过滤)——全空块发 {},不是省略。
        **({"observable": observable} if def_.observable is not None else {}),
        "resizable": def_.resizable,
        "hasDefaults": def_.defaults is not None,
        "hasCodegen": def_.frontend.codegen_key is not None,
    }
    if fields:
        out["fields"] = fields
    return out


def _render_frontend_def_ts(d) -> str:
    """FrontendNodeDef → generatedNodeSpecs 条目。defaults 三态:
    None → 不发 defaults 键(specFromGenerated 据此不造闭包,hasDefaults:false);
    ()/非空 → 按声明顺序全量渲染。
    resizable=False → 发 resizable: false(specFromGenerated 透传 resize)。"""
    lines = [f"  {_ts_str(d.type)}: {{"]
    lines.append(f"    label: {_ts_str(d.label)},")
    lines.append(f"    category: {_ts_str(d.category)},")
    if d.hint:
        lines.append(f"    hint: {_ts_str(d.hint)},")
    if d.family is not None:
        lines.append("    family: [" + ", ".join(_ts_str(f) for f in d.family) + "],")
    if d.defaults is not None:
        defaults_items = ", ".join(f"{_ts_str(k)}: {_ts_value(v)}" for k, v in d.defaults)
        lines.append(f"    defaults: {{ {defaults_items} }}," if d.defaults else "    defaults: {},")
    if d.fields:
        lines.append("    fields: [" + ", ".join(_render_field_ts(f) for f in d.fields) + "],")
    if d.observable is not None:
        ob = d.observable
        ob_parts = []
        if ob.info_markdown:
            ob_parts.append(f"infoMarkdown: {_ts_str(ob.info_markdown)}")
        if ob.viz_title:
            ob_parts.append(f"vizTitle: {_ts_str(ob.viz_title)}")
        if ob.viz_variant:
            ob_parts.append(f"vizVariant: {_ts_str(ob.viz_variant)}")
        if ob.spawns_viz_node is not None:
            ob_parts.append(f"spawnsVizNode: {_ts_value(ob.spawns_viz_node)}")
        # manifest 侧同律:块存在即发射,全空发 {}。
        lines.append("    observable: { " + ", ".join(ob_parts) + " }," if ob_parts else "    observable: {},")
    if not d.resizable:
        lines.append("    resizable: false,")
    if d.ui is not None:
        ui_parts = []
        if d.ui.accent:
            ui_parts.append(f"accent: {_ts_str(d.ui.accent)}")
        if d.ui.socket_rows:
            ui_parts.append(f"socketRows: {_ts_str(d.ui.socket_rows)}")
        if d.ui.code_kind:
            ui_parts.append(f"codeKind: {_ts_str(d.ui.code_kind)}")
        if d.ui.info_title is not None or d.ui.info_text is not None:
            ui_parts.append(
                "info: { title: " + _ts_str(d.ui.info_title or "") + ", text: " + _ts_str(d.ui.info_text or "") + " }"
            )
        lines.append("    ui: { " + ", ".join(ui_parts) + " },")
    ports_line = _render_ports_ts(getattr(d, "ports", None))
    if ports_line:
        lines.append(ports_line)
    fr = ["componentKey: " + _ts_str(d.frontend.component_key)]
    if d.frontend.codegen_key is not None:
        fr.append("codegenKey: " + _ts_str(d.frontend.codegen_key))
    if d.frontend.spec_code_key is not None:
        fr.append("specCodeKey: " + _ts_str(d.frontend.spec_code_key))
    lines.append("    frontend: { " + ", ".join(fr) + " },")
    lines.append("  },")
    return "\n".join(lines)


def _render_trainer_def_ts(d) -> str:
    return _render_loss_def_ts(d)


def _render_initialization_def_ts(d) -> str:
    return _render_loss_def_ts(d)


def _render_loss_def_ts(d) -> str:
    lines = [f"  {_ts_str(d.type)}: {{"]
    lines.append(f"    label: {_ts_str(d.label)},")
    lines.append(f"    category: {_ts_str(d.category)},")
    if d.hint:
        lines.append(f"    hint: {_ts_str(d.hint)},")
    if d.family is not None:
        lines.append("    family: [" + ", ".join(_ts_str(f) for f in d.family) + "],")
    defaults_items = ", ".join(f"{_ts_str(f.key)}: {_ts_value(f.default)}" for f in d.fields)
    lines.append(f"    defaults: {{ {defaults_items} }}," if d.fields else "    defaults: {},")
    if d.fields:
        lines.append("    fields: [" + ", ".join(_render_field_ts(f) for f in d.fields) + "],")
    fr = ["componentKey: " + _ts_str(d.frontend.component_key)]
    if d.frontend.codegen_key is not None:
        fr.append("codegenKey: " + _ts_str(d.frontend.codegen_key))
    lines.append("    frontend: { " + ", ".join(fr) + " },")
    lines.append("  },")
    return "\n".join(lines)


def _render_optimizer_def_ts(d) -> str:
    """OptimizerDef → generatedNodeSpecs 条目。SchemaNode 三面:富 fields
    (floatList/intList + positiveOnly/tooltip/ariaLabel)、ui 块、specCodeKey。"""
    lines = [f"  {_ts_str(d.type)}: {{"]
    lines.append(f"    label: {_ts_str(d.label)},")
    lines.append(f"    category: {_ts_str(d.category)},")
    if d.hint:
        lines.append(f"    hint: {_ts_str(d.hint)},")
    if d.family is not None:
        lines.append("    family: [" + ", ".join(_ts_str(f) for f in d.family) + "],")
    defaults_items = ", ".join(f"{_ts_str(f.key)}: {_ts_value(f.default)}" for f in d.fields)
    lines.append(f"    defaults: {{ {defaults_items} }}," if d.fields else "    defaults: {},")
    if d.fields:
        lines.append("    fields: [" + ", ".join(_render_field_ts(f) for f in d.fields) + "],")
    if d.ui is not None:
        ui_parts = []
        if d.ui.accent:
            ui_parts.append(f"accent: {_ts_str(d.ui.accent)}")
        if d.ui.socket_rows:
            ui_parts.append(f"socketRows: {_ts_str(d.ui.socket_rows)}")
        if d.ui.code_kind:
            ui_parts.append(f"codeKind: {_ts_str(d.ui.code_kind)}")
        if d.ui.info_title is not None or d.ui.info_text is not None:
            ui_parts.append(
                "info: { title: " + _ts_str(d.ui.info_title or "") + ", text: " + _ts_str(d.ui.info_text or "") + " }"
            )
        lines.append("    ui: { " + ", ".join(ui_parts) + " },")
    fr = ["componentKey: " + _ts_str(d.frontend.component_key)]
    if d.frontend.codegen_key is not None:
        fr.append("codegenKey: " + _ts_str(d.frontend.codegen_key))
    if d.frontend.spec_code_key is not None:
        fr.append("specCodeKey: " + _ts_str(d.frontend.spec_code_key))
    lines.append("    frontend: { " + ", ".join(fr) + " },")
    lines.append("  },")
    return "\n".join(lines)


def _render_model_def_ts(d) -> str:
    lines = [f"  {_ts_str(d.type)}: {{"]
    lines.append(f"    label: {_ts_str(d.label)},")
    lines.append(f"    category: {_ts_str(d.category)},")
    if d.hint:
        lines.append(f"    hint: {_ts_str(d.hint)},")
    if d.family is not None:
        lines.append("    family: [" + ", ".join(_ts_str(f) for f in d.family) + "],")
    if getattr(d, "spawn_defaults", None) is not None:
        # Preserve non-field runtime keys and their declared order.
        defaults_items = ", ".join(f"{_ts_str(k)}: {_ts_value(v)}" for k, v in d.spawn_defaults)
        lines.append(f"    defaults: {{ {defaults_items} }}," if d.spawn_defaults else "    defaults: {},")
    else:
        defaults_items = ", ".join(f"{_ts_str(f.key)}: {_ts_value(f.default)}" for f in d.fields)
        lines.append(f"    defaults: {{ {defaults_items} }}," if d.fields else "    defaults: {},")
    if d.fields:
        lines.append("    fields: [" + ", ".join(_render_field_ts(f) for f in d.fields) + "],")
    fr = ["componentKey: " + _ts_str(d.frontend.component_key)]
    if d.frontend.codegen_key is not None:
        fr.append("codegenKey: " + _ts_str(d.frontend.codegen_key))
    lines.append("    frontend: { " + ", ".join(fr) + " },")
    lines.append("  },")
    return "\n".join(lines)


def _render_dataset_def_ts(d) -> str:
    """DatasetDef → generatedNodeSpecs.ts 条目(UI 元数据全量;无 observable/spawn 块)。"""
    lines = [f"  {_ts_str(d.type)}: {{"]
    lines.append(f"    label: {_ts_str(d.label)},")
    lines.append(f"    category: {_ts_str(d.category)},")
    if d.hint:
        lines.append(f"    hint: {_ts_str(d.hint)},")
    if d.family is not None:
        lines.append("    family: [" + ", ".join(_ts_str(f) for f in d.family) + "],")
    defaults_items = ", ".join(f"{_ts_str(f.key)}: {_ts_value(f.default)}" for f in d.fields)
    lines.append(f"    defaults: {{ {defaults_items} }}," if d.fields else "    defaults: {},")
    if d.fields:
        lines.append("    fields: [" + ", ".join(_render_field_ts(f) for f in d.fields) + "],")
    ports_line = _render_ports_ts(getattr(d, "ports", None))
    if ports_line:
        lines.append(ports_line)
    fr = ["componentKey: " + _ts_str(d.frontend.component_key)]
    if d.frontend.codegen_key is not None:
        fr.append("codegenKey: " + _ts_str(d.frontend.codegen_key))
    lines.append("    frontend: { " + ", ".join(fr) + " },")
    lines.append("  },")
    return "\n".join(lines)


def render_generated_specs_ts_all(obs_defs, ds_defs, model_defs=(), opt_defs=(), loss_defs=(), init_defs=(), trainer_defs=(), frontend_defs=()) -> str:
    head = GENERATED_SPECS_HEADER + (
        'import type { GeneratedNodeSpec } from "../graph/generatedNodeSpecTypes";\n\n'
    )
    if not obs_defs and not ds_defs and not model_defs and not opt_defs and not loss_defs and not init_defs and not trainer_defs and not frontend_defs:
        return head + "export const GENERATED_NODE_SPECS: Record<string, GeneratedNodeSpec> = {};\n"
    rendered = {d.type: _render_def_ts(d) for d in obs_defs}
    rendered.update({d.type: _render_dataset_def_ts(d) for d in ds_defs})
    rendered.update({d.type: _render_model_def_ts(d) for d in model_defs})
    rendered.update({d.type: _render_optimizer_def_ts(d) for d in opt_defs})
    rendered.update({d.type: _render_loss_def_ts(d) for d in loss_defs})
    rendered.update({d.type: _render_initialization_def_ts(d) for d in init_defs})
    rendered.update({d.type: _render_trainer_def_ts(d) for d in trainer_defs})
    rendered.update({d.type: _render_frontend_def_ts(d) for d in frontend_defs})
    body = "\n".join(rendered[t] for t in sorted(rendered))
    return head + "export const GENERATED_NODE_SPECS: Record<string, GeneratedNodeSpec> = {\n" + body + "\n};\n"


def merge_and_render(fragment_text: str) -> dict[str, str]:
    legacy = parse_fragment(fragment_text)
    defs = registry.all_observable_defs()
    ds_defs = registry.all_dataset_defs()
    if defs or ds_defs:
        registry.validate_defs()
    # 三源两两不相交:observable defs vs legacy 在 merge_entries 内查;dataset defs
    # 与两者的重叠在此查(registry.dataset_def 已挡 defs 间同名,这里兜 legacy)。
    model_defs = registry.all_model_defs()
    entries = merge_entries(legacy, defs)
    if ds_defs:
        taken = {e["type"] for e in entries}
        ds_overlap = taken & {d.type for d in ds_defs}
        if ds_overlap:
            raise RuntimeError(
                f"node type(s) defined in BOTH channels (disjointness violated): {sorted(ds_overlap)}"
            )
        entries = entries + [def_to_entry_dataset(d) for d in ds_defs]
        entries.sort(key=lambda e: e["type"])
    if model_defs:
        taken = {e["type"] for e in entries}
        md_overlap = taken & {d.type for d in model_defs}
        if md_overlap:
            raise RuntimeError(
                f"node type(s) defined in BOTH channels (disjointness violated): {sorted(md_overlap)}"
            )
        entries = entries + [def_to_entry_model(d) for d in model_defs]
        entries.sort(key=lambda e: e["type"])
    opt_defs = registry.all_optimizer_defs()
    if opt_defs:
        taken = {e["type"] for e in entries}
        opt_overlap = taken & {d.type for d in opt_defs}
        if opt_overlap:
            raise RuntimeError(
                f"node type(s) defined in BOTH channels (disjointness violated): {sorted(opt_overlap)}"
            )
        entries = entries + [def_to_entry_optimizer(d) for d in opt_defs]
        entries.sort(key=lambda e: e["type"])
    loss_defs = registry.all_loss_defs()
    if loss_defs:
        taken = {e["type"] for e in entries}
        loss_overlap = taken & {d.type for d in loss_defs}
        if loss_overlap:
            raise RuntimeError(
                f"node type(s) defined in BOTH channels (disjointness violated): {sorted(loss_overlap)}"
            )
        entries = entries + [def_to_entry_loss(d) for d in loss_defs]
        entries.sort(key=lambda e: e["type"])
    init_defs = registry.all_initialization_defs()
    if init_defs:
        taken = {e["type"] for e in entries}
        init_overlap = taken & {d.type for d in init_defs}
        if init_overlap:
            raise RuntimeError(
                f"node type(s) defined in BOTH channels (disjointness violated): {sorted(init_overlap)}"
            )
        entries = entries + [def_to_entry_initialization(d) for d in init_defs]
        entries.sort(key=lambda e: e["type"])
    trainer_defs = registry.all_trainer_defs()
    if trainer_defs:
        taken = {e["type"] for e in entries}
        tr_overlap = taken & {d.type for d in trainer_defs}
        if tr_overlap:
            raise RuntimeError(
                f"node type(s) defined in BOTH channels (disjointness violated): {sorted(tr_overlap)}"
            )
        entries = entries + [def_to_entry_trainer(d) for d in trainer_defs]
        entries.sort(key=lambda e: e["type"])
    frontend_defs = registry.all_frontend_defs()
    if frontend_defs:
        taken = {e["type"] for e in entries}
        fe_overlap = taken & {d.type for d in frontend_defs}
        if fe_overlap:
            raise RuntimeError(
                f"node type(s) defined in BOTH channels (disjointness violated): {sorted(fe_overlap)}"
            )
        entries = entries + [def_to_entry_frontend(d) for d in frontend_defs]
        entries.sort(key=lambda e: e["type"])
    return {
        "manifest": serialize_manifest(entries),
        "node_kind": serialize_node_kind(entries),
        "capabilities": serialize_capabilities(entries),
        "param_models": serialize_param_models(entries),
        "generated_specs_ts": render_generated_specs_ts_all(defs, ds_defs, model_defs, opt_defs, loss_defs, init_defs, trainer_defs, frontend_defs),
    }


def main() -> int:
    args = sys.argv[1:]
    # Definitions provide the complete manifest, so the default fragment is empty.
    # Deprecated compatibility option for callers that still request fragments.
    if "--legacy-fragment" in args:
        fragment_path = Path(args[args.index("--legacy-fragment") + 1])
        fragment_text = fragment_path.read_text()
    else:
        fragment_text = "[]"
    rendered = merge_and_render(fragment_text)
    targets = [
        (OUT_FILES["manifest"], rendered["manifest"]),
        (OUT_FILES["node_kind"], rendered["node_kind"]),
        (OUT_FILES["capabilities"], rendered["capabilities"]),
        (OUT_FILES["param_models"], rendered["param_models"]),
        (OUT_FILES["generated_specs_ts"], rendered["generated_specs_ts"]),
    ]
    if "--check" in args:
        for path, content in targets:
            if not path.exists():
                print(f"Generated file is missing: {path}", file=sys.stderr)
                return 1
            if path.read_text() != content:
                print(f"Generated file is stale: {path}. Run npm run generate:node-manifest.", file=sys.stderr)
                return 1
        print("OK: node manifest, generated NodeKind, generated capabilities, generated param models, and generatedNodeSpecs.ts are current.")
        return 0
    for path, content in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(f"Wrote {len(parse_fragment(rendered['manifest']))} node manifest entries and generatedNodeSpecs.ts via merge generator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
