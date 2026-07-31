"""muon_optimizer — OptimizerDef-channel definition + builder provider.

程序化生成自 manifest;富 UI 元数据(tooltip/aria/min/max/positiveOnly)与
info 文案逐字转写自 OVERRIDES + optimizerNodeInfoContent.ts(python 持真相)。
SchemaNode 泛型渲染三面:fields/ui/specCode(SPEC_CODE_ADAPTERS)。
``build_optimizer_stage`` maps node metadata to the runtime configuration.
"""
from __future__ import annotations

from comfy_research.nodes.registry import optimizer_builder_for, optimizer_def
from comfy_research.nodes.schema import FloatListField, FrontendSpec, OptimizerDef, UiSpec

INFO_TEXT = 'Muon-style optimizer exposed by ComfyResearch runtime.\n\nThe node controls learning rate and momentum only; backend runtime defines the exact Muon update details.\n\nUse momentum near 0.9-0.98 for stable training on noisy objectives.\n\n**lr schedule** matches Adam/SGD: one socket for **lr_schedule** and **mup_lr_schedule** sources; muP grouping is not applied with Muon in this build.'

DEF = optimizer_def(
    OptimizerDef(
        type="muon_optimizer",
        label='Muon',
        hint="Muon-style optimizer with momentum for deeper-model experiments.",
        family=("optimizer_node",),
        fields=(
            FloatListField(key="learningRate", label='learning rate', default=0.003, positive_only=True, tooltip='e.g. 0.003 — comma-separated for multiple runs', aria_label='Muon learning rate'),
            FloatListField(key="momentum", label='momentum', default=0.95, min=0, max=1, aria_label='Muon momentum'),
        ),
        ui=UiSpec(
            accent="optimizer",
            socket_rows="optimizerLrSchedule",
            code_kind="optimizer",
            info_title='Muon optimizer',
            info_text=INFO_TEXT,
        ),
        frontend=FrontendSpec(component_key="SchemaNode", codegen_key="muon_optimizer", spec_code_key="muon_optimizer"),
    )
)


@optimizer_builder_for(DEF)
def build(model, config):
    from comfy_research.engine.optimizers.optimizer_builders import _build_muon_optimizer

    return _build_muon_optimizer(model, config)
