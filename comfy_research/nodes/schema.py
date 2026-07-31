"""Node definition schema for observables. Pure data, stdlib only.

Import rules (hard, guarded by AST test): this package must not runtime-import
ObservableRecorder, torch, or numpy — recorder signatures are typed via
TYPE_CHECKING; provider bodies keep heavy imports inside the function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Union

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

    RecorderFn = Callable[["ObservableRecorder", "Node"], None]
else:
    RecorderFn = Callable


@dataclass(frozen=True)
class IntField:
    key: str
    label: str
    default: int
    min: int | None = None
    max: int | None = None
    step: int | None = None
    # sweep 轴旋钮:False → axesFromGeneratedSpec 跳过(镜像字段/UI 隐藏字段)。
    sweepable: bool = True
    kind: str = "int"


@dataclass(frozen=True)
class FloatField:
    key: str
    label: str
    default: float
    min: float | None = None
    max: float | None = None
    step: float | None = None
    # sweep 轴旋钮:False → axesFromGeneratedSpec 跳过(镜像字段/UI 隐藏字段)。
    sweepable: bool = True
    # sweep 值语义:"int" → intChoices，用于运行时按整数解释的 float 字段。
    sweep_kind: str | None = None
    kind: str = "float"


@dataclass(frozen=True)
class EnumField:
    key: str
    label: str
    default: str
    options: tuple[str, ...] = ()
    # manifest 是否携带 options(loss 通道实况:mse 的 spec-level fields
    # 带 options 入 manifest;CE 的 defaults 推断字段不带——字节保真旋钮)。
    manifest_options: bool = False
    sweepable: bool = True
    kind: str = "enum"


@dataclass(frozen=True)
class BoolField:
    key: str
    label: str
    default: bool
    sweepable: bool = True
    kind: str = "boolean"


@dataclass(frozen=True)
class StringField:
    key: str
    label: str
    default: str
    sweepable: bool = False
    kind: str = "string"


# typing.Union (not X | Y) so the module also runs on the frontend CI's
# bare python3 (3.9): the wrapper invokes generate.py outside the conda env.
@dataclass(frozen=True)
class FloatListField:
    """optimizer 通道富 UI 字段(SchemaNode 泛型渲染;逗号分隔多值 sweep)。
    manifest 仍四键裸;positive_only/tooltip/aria_label 只进 generatedNodeSpecs。"""

    key: str
    label: str
    default: float
    min: float | None = None
    max: float | None = None
    positive_only: bool = False
    tooltip: str | None = None
    aria_label: str | None = None
    sweepable: bool = True
    kind: str = "floatList"


@dataclass(frozen=True)
class IntListField:
    key: str
    label: str
    default: int
    min: int | None = None
    positive_only: bool = False
    tooltip: str | None = None
    aria_label: str | None = None
    sweepable: bool = True
    kind: str = "intList"


Field = Union[IntField, FloatField, EnumField, BoolField, StringField, FloatListField, IntListField]


@dataclass(frozen=True)
class SpawnSpec:
    """Spawn defaultData recipe — exact per-node replication."""

    kind: str  # "hessian_topk" | "user_scalar" | "neuron_trajectory" | "scalar_series" | "embedding_trajectory" | "attention_map"
    fixed_top_k: int | None = None
    top_k_from_field: str | None = None
    unit: str | None = None
    # user_scalar 三态语义:unit=None → spawn 侧不传第四参
    # (builder 省略 vizYAxisLabel 键);要 "value" 就显式写 unit="value"。
    # title:spawn 出的 viz 头名与 vizTitle 不同时的逐 node 复刻;None → 用 vizTitle。
    title: str | None = None
    # 扩展:
    # order_from_field(hessian_topk):源 node data[key]=="ascending" 才升序,其余降序。
    order_from_field: str | None = None
    # series_labels(hessian_topk):spawn 数据附加 seriesLabels 数组(activation_stats)。
    series_labels: tuple[str, ...] | None = None
    # title_from_field(user_scalar/embedding_trajectory):data[key] trim 为标题,
    # 空值或缺失时回退到 title ?? vizTitle。
    title_from_field: str | None = None


@dataclass(frozen=True)
class PortAccept:
    """入口端口的一条接受规则。source_family 与 source_type 恰声明其一
    handles 非空 = 允许的 source handle。

    source_io_mode 仅当 readNodeCanvasIoMode(source.data) 等于该值时
    接受("model" | "input-output";None = 不限)。语义严格镜像 cascade 的
    fullModelModelSocket/fullModelTensorIoSource/combinedModel* 四谓词。
    刻意不做更泛的 data-predicate DSL。"""

    handles: tuple[str, ...]
    source_family: str | None = None
    source_type: str | None = None
    source_io_mode: str | None = None

    def __post_init__(self) -> None:
        if not self.handles:
            raise ValueError("PortAccept.handles must be non-empty")
        if (self.source_family is None) == (self.source_type is None):
            raise ValueError("PortAccept must declare exactly one of source_family/source_type")
        if self.source_io_mode not in (None, "model", "input-output"):
            raise ValueError(f"PortAccept.source_io_mode must be model/input-output, got {self.source_io_mode!r}")


@dataclass(frozen=True)
class InPort:
    """声明式入口端口:th==id 时按 accepts 判定;声明 ports 的节点为 return-style
    全权接管(未匹配任何 in-port 的 target handle → false,与原 cascade 分支一致)。"""

    id: str
    accepts: tuple[PortAccept, ...]

    def __post_init__(self) -> None:
        if not self.accepts:
            raise ValueError(f"InPort {self.id!r} must declare accepts")


@dataclass(frozen=True)
class VizSpec:
    variant: str
    title: str
    info_markdown: str
    spawns: bool = True
    user_whitelisted: bool = False
    spawn: SpawnSpec | None = None


@dataclass(frozen=True)
class FrontendSpec:
    component_key: str = "GenericObservableNode"
    codegen_key: str | None = None
    # SchemaNode show-code 面板的 specCode fn 键(TS SPEC_CODE_ADAPTERS)。
    spec_code_key: str | None = None


@dataclass(frozen=True)
class DatasetDef:
    """Dataset-channel node definition (NodeDef channel).

    独立于 ObservableDef:两通道的 manifest 字节形态不同——
    dataset manifest 字段恒为裸四键 {kind,key,label,defaultValue},
    options/min/max/step 只进 generatedNodeSpecs.ts(UI+sweep truth);observable
    通道的 options 会进 manifest。合用一个 emit 函数容易把 UI 元数据泄漏进
    dataset manifest,破坏字节恒等。
    """

    type: str
    label: str
    # family 逐字节复刻现 manifest(node_capabilities.py 由它派生;materialize
    # 的 capability 分派依赖这一点)。None 表示 manifest 无 family 键。
    family: tuple[str, ...] | None
    fields: tuple[Field, ...] = ()
    hint: str | None = None  # 46 个存量 dataset 全无 hint;新增 node 可用
    frontend: FrontendSpec = field(
        default_factory=lambda: FrontendSpec(component_key="GenericDatasetNode")
    )
    category: str = "dataset"
    # 启用:声明式入口端口(dataset_mixer/_b、input_sampler 首批)。
    ports: tuple[InPort, ...] | None = None


@dataclass(frozen=True)
class ModelDef:
    """Model-channel node definition (NodeDef channel)。

    MODEL_BUILDERS members register model_builder_for with the ModelBuilder
    signature. Atomic-chain and tensor/canvas helper nodes have no provider;
    atomic_layer_chain dispatches atomic nodes. Fields are inferred from defaults;
    options/min/max/
    sweepable 只进 generatedNodeSpecs(dataset 同款字节纪律)。
    """

    type: str
    label: str
    family: tuple[str, ...] | None
    fields: tuple[Field, ...] = ()
    hint: str | None = None
    frontend: FrontendSpec = field(default_factory=FrontendSpec)
    category: str = "model"
    # tensor 族 spawn defaults 覆盖包含非 field 运行态键
    # (outputTensor/lastError: null、shape: list、exceptDim: null),fields 无法
    # 表达。非 None 时 generatedNodeSpecs 的 defaults 按此**有序全量**渲染
    # (键序保持显式声明顺序);manifest fields/hasDefaults 不受影响。
    spawn_defaults: tuple[tuple[str, object], ...] | None = None
    # Reserved declaration slot for model initialization sockets.
    ports: None = None


@dataclass(frozen=True)
class TrainerDef:
    """Trainer-channel node definition。两种 trainer 均无 provider：
    trainer 的 runtime 是 prepare pipeline 本身；custom 组件提供 run 按钮和
    进度 UI。computeDevice sweepable=False，因此不生成 sweep 轴；
    boolean 字段天然无轴(axesFromGeneratedSpec 无 boolean 分支)。"""

    type: str
    label: str
    family: tuple[str, ...] | None = None
    fields: tuple[Field, ...] = ()
    hint: str | None = None
    frontend: FrontendSpec = field(default_factory=FrontendSpec)
    category: str = "training"


@dataclass(frozen=True)
class InitializationDef:
    """Initialization-channel node definition。三型全部零 provider:
    apply 分派、init socket 连接规则和 supported 集由 trainer runtime 与
    canvas 实现；这些节点不生成 sweep 轴。"""

    type: str
    label: str
    family: tuple[str, ...] | None = None
    fields: tuple[Field, ...] = ()
    hint: str | None = None
    frontend: FrontendSpec = field(default_factory=FrontendSpec)
    category: str = "analysis"


@dataclass(frozen=True)
class LossDef:
    """Loss-channel node definition.

    Primary criteria may provide ``loss_criterion_for``; auxiliary loss-socket
    nodes are evaluated by ``loss_terms``; ``kan_reg`` is an ObservableDef.
    """

    type: str
    label: str
    family: tuple[str, ...] | None
    fields: tuple[Field, ...] = ()
    hint: str | None = None
    frontend: FrontendSpec = field(default_factory=FrontendSpec)
    category: str = "loss"


@dataclass(frozen=True)
class UiSpec:
    """SchemaNode 泛型 UI 块；info 文案由 Python definition 提供。"""

    accent: str | None = None
    socket_rows: str | None = None
    code_kind: str | None = None
    info_title: str | None = None
    info_text: str | None = None


@dataclass(frozen=True)
class OptimizerDef:
    """Optimizer-channel node definition.

    Seven optimizer nodes use the generic SchemaNode fields, UI, and spec-code
    adapters. Schedule nodes keep custom components. Runtime builders register
    through optimizer_builder_for and receive OptimizerBuildConfig.
    """

    type: str
    label: str
    family: tuple[str, ...] | None
    fields: tuple[Field, ...] = ()
    hint: str | None = None
    frontend: FrontendSpec = field(default_factory=FrontendSpec)
    ui: UiSpec | None = None
    category: str = "optimizer"


@dataclass(frozen=True)
class FrontendObservableBlock:
    """前端节点的 manifest observable 元数据块(observable_viz 的
    infoMarkdown 等;与 ObservableDef.viz 不同——无 recorder/spawn 语义,纯转写)。"""

    info_markdown: str | None = None
    viz_title: str | None = None
    viz_variant: str | None = None
    spawns_viz_node: bool | None = None


# 前端节点的 category 封闭集。
FRONTEND_NODE_CATEGORIES = frozenset(
    {"analysis", "visualization", "language", "internal", "checkpoint", "training"}
)


@dataclass(frozen=True)
class FrontendNodeDef:
    """第八通道——纯前端 UI/analysis 节点(零后端 runtime:无 provider/
    recorder/builder,orphan 守卫按此豁免)。

    defaults 三态:None → manifest hasDefaults:false 且
    generatedNodeSpecs 不发 defaults 键;() → 空 defaults {}(emptyDefaults 型);
    非空 → 有序全量 spawn data(键序保持显式声明顺序,值支持
    标量/None/list/dict——tensor_selector 的 activationTensorCaches: {} 实况)。
    fields 是 defaults 的标量子集(inferFieldsFromDefaults 推断链的显式转写),
    __post_init__ 钉 fields ⊆ defaults 且值逐键相等。"""

    type: str
    label: str
    category: str
    hint: str | None = None
    family: tuple[str, ...] | None = None
    fields: tuple[Field, ...] = ()
    defaults: tuple[tuple[str, object], ...] | None = None
    observable: FrontendObservableBlock | None = None
    resizable: bool = True
    frontend: FrontendSpec = field(default_factory=FrontendSpec)
    ui: UiSpec | None = None
    ports: tuple[InPort, ...] | None = None

    def __post_init__(self) -> None:
        if self.category not in FRONTEND_NODE_CATEGORIES:
            raise ValueError(
                f"FrontendNodeDef {self.type!r}: category {self.category!r} not in {sorted(FRONTEND_NODE_CATEGORIES)}"
            )
        keys = [f.key for f in self.fields]
        if len(keys) != len(set(keys)):
            raise ValueError(f"FrontendNodeDef {self.type!r}: duplicate field keys")
        if self.defaults is None and self.fields:
            # 无 defaults 却有 fields = hasDefaults:false + param fields 怪状态。
            raise ValueError(f"FrontendNodeDef {self.type!r}: fields require defaults")
        if self.defaults is not None:
            dd = dict(self.defaults)
            if len(dd) != len(self.defaults):
                raise ValueError(f"FrontendNodeDef {self.type!r}: duplicate defaults keys")
            for f in self.fields:
                if f.key not in dd or dd[f.key] != f.default:
                    raise ValueError(
                        f"FrontendNodeDef {self.type!r}: field {f.key!r} must mirror defaults value"
                    )


@dataclass(frozen=True)
class ObservableDef:
    type: str
    label: str
    viz: VizSpec
    # hint 可选:OVERRIDES 系 node 的 manifest 条目没有 hint 键;None 时
    # manifest 与 generatedNodeSpecs 都不发射该 key(buildManifest truthy 语义)。
    hint: str | None = None
    fields: tuple[Field, ...] = ()
    frontend: FrontendSpec = field(default_factory=FrontendSpec)
    category: str = "observables"
    # 启用:声明式入口端口(observable_user 首批)。
    ports: tuple[InPort, ...] | None = None
    connection_policy: None = None
