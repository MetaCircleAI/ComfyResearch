"""signsgd_optimizer — OptimizerDef-channel definition + builder provider.

程序化生成自 manifest;富 UI 元数据(tooltip/aria/min/max/positiveOnly)与
info 文案逐字转写自 OVERRIDES + optimizerNodeInfoContent.ts(python 持真相)。
SchemaNode 泛型渲染三面:fields/ui/specCode(SPEC_CODE_ADAPTERS)。
``build_optimizer_stage`` maps node metadata to the runtime configuration.
"""
from __future__ import annotations

from comfy_research.nodes.registry import optimizer_builder_for, optimizer_def
from comfy_research.nodes.schema import FloatListField, FrontendSpec, OptimizerDef, UiSpec

INFO_TEXT = 'SignSGD updates parameters by the sign of each gradient component.\n\nThis node controls learning rate and optional L2 weight decay before taking the gradient sign.\n\nIt is useful for optimizer-comparison experiments where update direction, rather than gradient magnitude, is the main variable.\n\nOptional **lr schedule** input: one socket, matching Adam/SGD. muP grouping is not applied with SignSGD in this build.'

DEF = optimizer_def(
    OptimizerDef(
        type="signsgd_optimizer",
        label='SignSGD',
        hint="SignSGD optimizer that steps by gradient sign.",
        family=("optimizer_node",),
        fields=(
            FloatListField(key="learningRate", label='learning rate', default=0.001, positive_only=True, tooltip='e.g. 0.001, comma-separated for multiple runs', aria_label='SignSGD learning rate'),
            FloatListField(key="weightDecay", label='weight decay', default=0, min=0, tooltip='L2 penalty coefficient added before taking the gradient sign', aria_label='SignSGD weight decay'),
        ),
        ui=UiSpec(
            accent="optimizer",
            socket_rows="optimizerLrSchedule",
            code_kind="optimizer",
            info_title='SignSGD optimizer',
            info_text=INFO_TEXT,
        ),
        frontend=FrontendSpec(component_key="SchemaNode", codegen_key="signsgd_optimizer", spec_code_key="signsgd_optimizer"),
    )
)


@optimizer_builder_for(DEF)
def build(model, config):
    from comfy_research.engine.optimizers.optimizer_builders import _build_signsgd_optimizer

    return _build_signsgd_optimizer(model, config)
